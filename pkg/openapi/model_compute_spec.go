// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Flame REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

// ComputeSpec - Compute specification
type ComputeSpec struct {
	DeployerId string `json:"deployerId"`

	AdminId string `json:"adminId,omitempty"`

	Region string `json:"region,omitempty"`

	ComputeId string `json:"computeId,omitempty"`

	ApiKey string `json:"apiKey,omitempty"`
}

// AssertComputeSpecRequired checks if the required fields are not zero-ed
func AssertComputeSpecRequired(obj ComputeSpec) error {
	elements := map[string]interface{}{
		"deployerId": obj.DeployerId,
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return &RequiredError{Field: name}
		}
	}

	return nil
}

// AssertRecurseComputeSpecRequired recursively checks if required fields are not zero-ed in a nested slice.
// Accepts only nested slice of ComputeSpec (e.g. [][]ComputeSpec), otherwise ErrTypeAssertionError is thrown.
func AssertRecurseComputeSpecRequired(objSlice interface{}) error {
	return AssertRecurseInterfaceRequired(objSlice, func(obj interface{}) error {
		aComputeSpec, ok := obj.(ComputeSpec)
		if !ok {
			return ErrTypeAssertionError
		}
		return AssertComputeSpecRequired(aComputeSpec)
	})
}
