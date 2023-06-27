# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""SCAFFOLD horizontal FL trainer."""

import logging
import time

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import TrainState, DeviceType
from flame.common.custom_abcmeta import abstract_attribute
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    weights_to_device,
    weights_to_model_device,
)
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

from flame.mode.horizontal.trainer import Trainer as BaseTrainer

logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"
TAG_UPLOAD_DATASET_SIZE = "uploadDatasetSize"

class Trainer(BaseTrainer):
    """Trainer implements an ML training role."""
    
    @abstract_attribute
    def learning_rate(self):
        """Abstract attribute for learning rate."""
    
    @abstract_attribute
    def batch_size(self):
        """Abstract attribute for batch size."""
    
    @abstract_attribute
    def epochs(self):
        """Abstract attribute for epochs."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        logger.debug("in internal_init")
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for scaffold) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().internal_init()

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if not msg:
            logger.debug("no message received")
            if self._work_done:
                # when the work is done, we cancel continue condition
                # (i.e., we set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()
            self.regularizer.save_state(TrainState.PRE, glob_model=self.model)
        
        # trainer needs weight prior to setting c-term for regularizer
        if MessageType.CLIENT_WEIGHT in msg:
            self.regularizer.client_weight = msg[MessageType.CLIENT_WEIGHT]
        
        if MessageType.C_WEIGHTS in msg:
            self.regularizer.set_c_glob(weights_to_model_device(msg[MessageType.C_WEIGHTS], self.model))

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        if MessageType.DATASAMPLER_METADATA in msg:
            self.datasampler.handle_metadata_from_aggregator(
                msg[MessageType.DATASAMPLER_METADATA]
            )        
        
        self.fetch_success = True
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        elif tag == TAG_UPLOAD_DATASET_SIZE:
            self._send_dataset_size(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        self._update_weights()
        self.regularizer.save_state(TrainState.POST, loc_model=self.model)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        delta_weights = self.privacy.apply_dp_fn(delta_weights)
        
        # provide necessary information for regularizer to update c-terms
        self.regularizer.learning_rate = self.learning_rate
        self.regularizer.batch_size = self.batch_size
        self.regularizer.epochs = self.epochs
        self.regularizer.dataset_size = self.dataset_size
        self.regularizer.update()

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.C_WEIGHTS: weights_to_device(self.regularizer.delta_c_loc, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
        }
        channel.send(end, msg)
        logger.debug("sending weights done")
    
    def _send_dataset_size(self, tag: str) -> None:
        logger.debug("calling _send_dataset_size")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_dataset_size] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        msg = {MessageType.DATASET_SIZE: self.dataset_size}
        channel.send(end, msg)
        logger.debug("sending dataset size done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)
            
            task_put_dataset_size = Tasklet("upload_dataset_size", self.put, TAG_UPLOAD_DATASET_SIZE)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_put_dataset_size >> task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_UPLOAD_DATASET_SIZE]
