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

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/gorilla/mux"
)

// ComputesApiController binds http requests to an api service and writes the service results to the http response
type ComputesApiController struct {
	service      ComputesApiServicer
	errorHandler ErrorHandler
}

// ComputesApiOption for how the controller is set up.
type ComputesApiOption func(*ComputesApiController)

// WithComputesApiErrorHandler inject ErrorHandler into controller
func WithComputesApiErrorHandler(h ErrorHandler) ComputesApiOption {
	return func(c *ComputesApiController) {
		c.errorHandler = h
	}
}

// NewComputesApiController creates a default api controller
func NewComputesApiController(s ComputesApiServicer, opts ...ComputesApiOption) Router {
	controller := &ComputesApiController{
		service:      s,
		errorHandler: DefaultErrorHandler,
	}

	for _, opt := range opts {
		opt(controller)
	}

	return controller
}

// Routes returns all the api routes for the ComputesApiController
func (c *ComputesApiController) Routes() Routes {
	return Routes{
		{
			"DeleteCompute",
			strings.ToUpper("Delete"),
			"/computes/{computeId}",
			c.DeleteCompute,
		},
		{
			"GetAllComputes",
			strings.ToUpper("Get"),
			"/computes",
			c.GetAllComputes,
		},
		{
			"GetComputeConfig",
			strings.ToUpper("Get"),
			"/computes/{computeId}/config",
			c.GetComputeConfig,
		},
		{
			"GetComputeStatus",
			strings.ToUpper("Get"),
			"/computes/{computeId}",
			c.GetComputeStatus,
		},
		{
			"GetDeploymentConfig",
			strings.ToUpper("Get"),
			"/computes/{computeId}/deployments/{jobId}/config",
			c.GetDeploymentConfig,
		},
		{
			"GetDeploymentStatus",
			strings.ToUpper("Get"),
			"/computes/{computeId}/deployments/{jobId}/status",
			c.GetDeploymentStatus,
		},
		{
			"GetDeployments",
			strings.ToUpper("Get"),
			"/computes/{computeId}/deployments",
			c.GetDeployments,
		},
		{
			"PutDeploymentStatus",
			strings.ToUpper("Put"),
			"/computes/{computeId}/deployments/{jobId}/status",
			c.PutDeploymentStatus,
		},
		{
			"RegisterCompute",
			strings.ToUpper("Post"),
			"/computes",
			c.RegisterCompute,
		},
		{
			"UpdateCompute",
			strings.ToUpper("Put"),
			"/computes/{computeId}",
			c.UpdateCompute,
		},
	}
}

// DeleteCompute - Delete compute cluster specification
func (c *ComputesApiController) DeleteCompute(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.DeleteCompute(r.Context(), computeIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetAllComputes - Get all computes owned by an admin
func (c *ComputesApiController) GetAllComputes(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	adminIdParam := query.Get(constants.ParamAdminID)

	result, err := c.service.GetAllComputes(r.Context(), adminIdParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetComputeConfig - Get configuration for a compute cluster
func (c *ComputesApiController) GetComputeConfig(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.GetComputeConfig(r.Context(), computeIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetComputeStatus - Get status of a given compute cluster
func (c *ComputesApiController) GetComputeStatus(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.GetComputeStatus(r.Context(), computeIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetDeploymentConfig - Get the deployment config for a job for a compute cluster
func (c *ComputesApiController) GetDeploymentConfig(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	jobIdParam := params[constants.ParamJobID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.GetDeploymentConfig(r.Context(), computeIdParam, jobIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetDeploymentStatus - Get the deployment status for a job on a compute cluster
func (c *ComputesApiController) GetDeploymentStatus(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	jobIdParam := params[constants.ParamJobID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.GetDeploymentStatus(r.Context(), computeIdParam, jobIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetDeployments - Get all deployments within a compute cluster
func (c *ComputesApiController) GetDeployments(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	result, err := c.service.GetDeployments(r.Context(), computeIdParam, xAPIKEYParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// PutDeploymentStatus - Add or update the deployment status for a job on a compute cluster
func (c *ComputesApiController) PutDeploymentStatus(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	jobIdParam := params[constants.ParamJobID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	requestBodyParam := map[string]AgentState{}
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&requestBodyParam); err != nil {
		c.errorHandler(w, r, &ParsingError{Err: err}, nil)
		return
	}
	result, err := c.service.PutDeploymentStatus(r.Context(), computeIdParam, jobIdParam, xAPIKEYParam, requestBodyParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// RegisterCompute - Register a new compute cluster
func (c *ComputesApiController) RegisterCompute(w http.ResponseWriter, r *http.Request) {
	computeSpecParam := ComputeSpec{}
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&computeSpecParam); err != nil {
		c.errorHandler(w, r, &ParsingError{Err: err}, nil)
		return
	}
	if err := AssertComputeSpecRequired(computeSpecParam); err != nil {
		c.errorHandler(w, r, err, nil)
		return
	}
	result, err := c.service.RegisterCompute(r.Context(), computeSpecParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// UpdateCompute - Update a compute cluster's specification
func (c *ComputesApiController) UpdateCompute(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	computeIdParam := params[constants.ParamComputeID]

	xAPIKEYParam := r.Header.Get("X-API-KEY")
	computeSpecParam := ComputeSpec{}
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&computeSpecParam); err != nil {
		c.errorHandler(w, r, &ParsingError{Err: err}, nil)
		return
	}
	if err := AssertComputeSpecRequired(computeSpecParam); err != nil {
		c.errorHandler(w, r, err, nil)
		return
	}
	result, err := c.service.UpdateCompute(r.Context(), computeIdParam, xAPIKEYParam, computeSpecParam)
	// If an error occurred, encode the error with the status code
	if err != nil {
		c.errorHandler(w, r, err, &result)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}
