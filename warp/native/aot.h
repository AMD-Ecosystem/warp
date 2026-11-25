// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Warp AOT (Ahead-Of-Time) utilities for C++ integration
// This header provides common boilerplate for C++ programs that use
// Warp-generated kernels via PTX loading or source inclusion.

// Auto-detect CUDA compilation (HIP compilation for AMD GPUs counts as CUDA here)
#if defined(__CUDACC__) || defined(__CUDA__) || defined(__NVCC__) || defined(__HIP__)
#define WP_ENABLE_CUDA 1
#else
#define WP_ENABLE_CUDA 0
#endif

// Default tile block dimension (can be overridden before including aot.h)
#ifndef WP_TILE_BLOCK_DIM
#define WP_TILE_BLOCK_DIM 256
#endif

// CUDA-specific includes and error checking macros
#if WP_ENABLE_CUDA
#include <cstdlib>  // For exit() in error checking macros
#include <iostream>  // For std::cerr in error checking macros

#if defined(__HIP__)
#include <hip/hip_runtime.h>  // HIP Runtime API, which also provides the Driver API functions

// The CUDA names used by Warp's C++ examples, mapped to HIP
#define CUcontext hipCtx_t
#define CUdevice hipDevice_t
#define CUfunction hipFunction_t
#define CUmodule hipModule_t
#define CUresult hipError_t
#define CUDA_SUCCESS hipSuccess
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetName hipDeviceGetName
#define cuDevicePrimaryCtxRelease hipDevicePrimaryCtxRelease
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuGetErrorString hipDrvGetErrorString
#define cuInit hipInit
#define cuLaunchKernel hipModuleLaunchKernel
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleLoadData hipModuleLoadData
#define cuModuleUnload hipModuleUnload
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphLaunch hipGraphLaunch
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#else
#include <cuda.h>  // CUDA Driver API
#include <cuda_runtime.h>  // CUDA Runtime API
#endif

#ifndef CHECK_CU
// CUDA Driver API error checking
#define CHECK_CU(call) \
        do { \
            CUresult err = call; \
            if (err != CUDA_SUCCESS) { \
                const char* errStr; \
                cuGetErrorString(err, &errStr); \
                std::cerr << "CUDA Driver API error at " << __FILE__ << ":" << __LINE__ \
                          << " - " << errStr << std::endl; \
                exit(1); \
            } \
        } while(0)
#endif

#ifndef CHECK_CUDA
// CUDA Runtime API error checking
#define CHECK_CUDA(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA Runtime API error at " << __FILE__ << ":" << __LINE__ \
                          << " - " << cudaGetErrorString(err) << std::endl; \
                exit(1); \
            } \
        } while(0)
#endif

#endif  // WP_ENABLE_CUDA

// Include Warp builtin types and functions
#include "builtin.h"
