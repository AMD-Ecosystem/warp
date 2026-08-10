// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if WP_ENABLE_CUDA

#include "cuda_util.h"
#include "error.h"

// HIP headers are already included via cuda_util.h -> hip_util.h
#if !defined(__HIP_PLATFORM_AMD__)
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wingdi.h>  // needed for OpenGL includes
#elif defined(__linux__)
#include <dlfcn.h>
#endif
#endif  // !__HIP_PLATFORM_AMD__

#include <queue>
#include <unordered_set>
#include <vector>
#include <mutex>

#if defined(__HIP_PLATFORM_AMD__)
namespace {
std::once_flag g_hip_context_init;
std::vector<HipContext> g_hip_contexts;
thread_local std::vector<HipContext*> g_hip_context_stack;

void init_hip_contexts()
{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0)
        return;

    g_hip_contexts.resize(count);
    for (int i = 0; i < count; ++i)
        g_hip_contexts[i].device = i;
}

HipContext* get_hip_context_for_device(int device)
{
    std::call_once(g_hip_context_init, init_hip_contexts);
    if (device < 0 || device >= static_cast<int>(g_hip_contexts.size()))
        return nullptr;
    return &g_hip_contexts[device];
}
}  // namespace
#endif  // defined(__HIP_PLATFORM_AMD__)

#if defined(__HIP_PLATFORM_AMD__)
#if HIP_VERSION > 0
#define WP_HIP_VERSION ((HIP_VERSION / 10000000) * 100 + ((HIP_VERSION / 100000) % 100))
#else
#define WP_HIP_VERSION 0
#endif  // HIP_VERSION > 0
#endif  // __HIP_PLATFORM_AMD__
 
#if defined(__HIP_PLATFORM_AMD__)
#define WP_DRIVER_ENTRY_VERSION(_version) (WP_HIP_VERSION)
#else
#define WP_DRIVER_ENTRY_VERSION(_version) (_version)
#endif  // __HIP_PLATFORM_AMD__
 
// the minimum CUDA version required from the driver
#define WP_CUDA_DRIVER_VERSION 12000

// the minimum CUDA Toolkit version required to build Warp
#define WP_CUDA_TOOLKIT_VERSION 12000

// check if the CUDA Toolkit (or ROCm) is too old
#if defined(__HIP_PLATFORM_AMD__)
#ifndef WP_HIP_MIN_VERSION
#define WP_HIP_MIN_VERSION 70000000
#endif  // WP_HIP_MIN_VERSION
#if CUDA_VERSION < WP_HIP_MIN_VERSION
#error Building Warp requires ROCm 7.0 or higher
#endif  // CUDA_VERSION < WP_HIP_MIN_VERSION
#else
#if CUDA_VERSION < WP_CUDA_TOOLKIT_VERSION
#error Building Warp requires CUDA Toolkit version 12.0 or higher
#endif
#endif  // __HIP_PLATFORM_AMD__

// Avoid including <cudaGLTypedefs.h>, which requires OpenGL headers to be installed.
// We define our own GL types, based on the spec here: https://www.khronos.org/opengl/wiki/OpenGL_Type
namespace wp {
typedef uint32_t GLuint;
typedef uint32_t GLenum;
}

// function prototypes adapted from <cudaGLTypedefs.h>
typedef CUresult(CUDAAPI* PFN_cuGraphicsGLRegisterBuffer_v3000)(
    CUgraphicsResource* pCudaResource, wp::GLuint buffer, unsigned int Flags
);
typedef CUresult(CUDAAPI* PFN_cuGraphicsGLRegisterImage_v3000)(
    CUgraphicsResource* pCudaResource, wp::GLuint image, wp::GLenum target, unsigned int Flags
);

// function prototypes adapted from <cudaProfilerTypedefs.h>. We declare these locally to avoid
// including the header, which pulls in <cudaProfiler.h>; that header ships only in the separate
// cuda_profiler_api redist component, not the cuda_cudart component used by the builder images.
typedef CUresult(CUDAAPI* PFN_cuProfilerStart_v4000)(void);
typedef CUresult(CUDAAPI* PFN_cuProfilerStop_v4000)(void);


// function pointers to driver API entry points
// these are explicitly versioned according to cudaTypedefs.h from CUDA Toolkit WP_CUDA_TOOLKIT_VERSION

#if !defined(__HIP_PLATFORM_AMD__)
#if CUDA_VERSION >= 13000
#define PFN_cuGetProcAddress  PFN_cuGetProcAddress_v12000
#endif

static PFN_cuGetProcAddress_v12000 pfn_cuGetProcAddress;
static PFN_cuDriverGetVersion_v2020 pfn_cuDriverGetVersion;
static PFN_cuGetErrorName_v6000 pfn_cuGetErrorName;
static PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;
static PFN_cuInit_v2000 pfn_cuInit;
static PFN_cuDeviceGet_v2000 pfn_cuDeviceGet;
static PFN_cuDeviceGetCount_v2000 pfn_cuDeviceGetCount;
static PFN_cuDeviceGetName_v2000 pfn_cuDeviceGetName;
static PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute;
static PFN_cuDeviceGetUuid_v11040 pfn_cuDeviceGetUuid;
static PFN_cuDevicePrimaryCtxRetain_v7000 pfn_cuDevicePrimaryCtxRetain;
static PFN_cuDevicePrimaryCtxRelease_v11000 pfn_cuDevicePrimaryCtxRelease;
static PFN_cuDeviceCanAccessPeer_v4000 pfn_cuDeviceCanAccessPeer;
static PFN_cuMemGetInfo_v3020 pfn_cuMemGetInfo;
#if CUDA_VERSION >= 12080
static PFN_cuMemcpyBatchAsync_v12080 pfn_cuMemcpyBatchAsync;
#endif
static PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent;
static PFN_cuCtxSetCurrent_v4000 pfn_cuCtxSetCurrent;
static PFN_cuCtxPushCurrent_v4000 pfn_cuCtxPushCurrent;
static PFN_cuCtxPopCurrent_v4000 pfn_cuCtxPopCurrent;
static PFN_cuCtxSynchronize_v2000 pfn_cuCtxSynchronize;
static PFN_cuCtxGetDevice_v2000 pfn_cuCtxGetDevice;
static PFN_cuCtxCreate_v3020 pfn_cuCtxCreate;
static PFN_cuCtxDestroy_v4000 pfn_cuCtxDestroy;
static PFN_cuCtxEnablePeerAccess_v4000 pfn_cuCtxEnablePeerAccess;
static PFN_cuCtxDisablePeerAccess_v4000 pfn_cuCtxDisablePeerAccess;
static PFN_cuStreamCreate_v2000 pfn_cuStreamCreate;
static PFN_cuStreamDestroy_v4000 pfn_cuStreamDestroy;
static PFN_cuStreamQuery_v2000 pfn_cuStreamQuery;
static PFN_cuStreamSynchronize_v2000 pfn_cuStreamSynchronize;
static PFN_cuStreamWaitEvent_v3020 pfn_cuStreamWaitEvent;
static PFN_cuStreamGetCtx_v9020 pfn_cuStreamGetCtx;
static PFN_cuStreamGetCaptureInfo_v11030 pfn_cuStreamGetCaptureInfo;
static PFN_cuStreamUpdateCaptureDependencies_v11030 pfn_cuStreamUpdateCaptureDependencies;
static PFN_cuStreamCreateWithPriority_v5050 pfn_cuStreamCreateWithPriority;
static PFN_cuStreamGetPriority_v5050 pfn_cuStreamGetPriority;
static PFN_cuEventCreate_v2000 pfn_cuEventCreate;
static PFN_cuEventDestroy_v4000 pfn_cuEventDestroy;
static PFN_cuEventQuery_v2000 pfn_cuEventQuery;
static PFN_cuEventRecord_v2000 pfn_cuEventRecord;
static PFN_cuEventRecordWithFlags_v11010 pfn_cuEventRecordWithFlags;
static PFN_cuEventSynchronize_v2000 pfn_cuEventSynchronize;
#if CUDA_VERSION >= 12030
// function used to add conditional graph nodes, not available in older CUDA versions
static PFN_cuGraphAddNode_v12030 pfn_cuGraphAddNode;
#endif
static PFN_cuGraphNodeGetDependentNodes_v10000 pfn_cuGraphNodeGetDependentNodes;
static PFN_cuGraphNodeGetType_v10000 pfn_cuGraphNodeGetType;
static PFN_cuModuleLoadDataEx_v2010 pfn_cuModuleLoadDataEx;
static PFN_cuModuleUnload_v2000 pfn_cuModuleUnload;
static PFN_cuModuleGetFunction_v2000 pfn_cuModuleGetFunction;
static PFN_cuLaunchKernel_v4000 pfn_cuLaunchKernel;
static PFN_cuOccupancyMaxPotentialBlockSize_v6050 pfn_cuOccupancyMaxPotentialBlockSize;
static PFN_cuOccupancyMaxActiveClusters_v11070 pfn_cuOccupancyMaxActiveClusters;
static PFN_cuMemcpyPeerAsync_v4000 pfn_cuMemcpyPeerAsync;
static PFN_cuPointerGetAttribute_v4000 pfn_cuPointerGetAttribute;
static PFN_cuGraphicsMapResources_v3000 pfn_cuGraphicsMapResources;
static PFN_cuGraphicsUnmapResources_v3000 pfn_cuGraphicsUnmapResources;
static PFN_cuGraphicsResourceGetMappedPointer_v3020 pfn_cuGraphicsResourceGetMappedPointer;
static PFN_cuGraphicsGLRegisterBuffer_v3000 pfn_cuGraphicsGLRegisterBuffer;
static PFN_cuGraphicsGLRegisterImage_v3000 pfn_cuGraphicsGLRegisterImage;
static PFN_cuGraphicsSubResourceGetMappedArray_v3000 pfn_cuGraphicsSubResourceGetMappedArray;
static PFN_cuGraphicsUnregisterResource_v3000 pfn_cuGraphicsUnregisterResource;
static PFN_cuModuleGetGlobal_v3020 pfn_cuModuleGetGlobal;
static PFN_cuFuncSetAttribute_v9000 pfn_cuFuncSetAttribute;
static PFN_cuFuncGetAttribute_v2020 pfn_cuFuncGetAttribute;
static PFN_cuIpcGetEventHandle_v4010 pfn_cuIpcGetEventHandle;
static PFN_cuIpcOpenEventHandle_v4010 pfn_cuIpcOpenEventHandle;
static PFN_cuIpcGetMemHandle_v4010 pfn_cuIpcGetMemHandle;
static PFN_cuIpcOpenMemHandle_v11000 pfn_cuIpcOpenMemHandle;
static PFN_cuIpcCloseMemHandle_v4010 pfn_cuIpcCloseMemHandle;

// Profiler control functions
static PFN_cuProfilerStart_v4000 pfn_cuProfilerStart;
static PFN_cuProfilerStop_v4000 pfn_cuProfilerStop;

// Texture functions
static PFN_cuArrayCreate_v3020 pfn_cuArrayCreate;
static PFN_cuArrayDestroy_v2000 pfn_cuArrayDestroy;
static PFN_cuArray3DCreate_v3020 pfn_cuArray3DCreate;
static PFN_cuArray3DGetDescriptor_v3020 pfn_cuArray3DGetDescriptor;
static PFN_cuMemcpy2D_v3020 pfn_cuMemcpy2D;
static PFN_cuMemcpy2DAsync_v3020 pfn_cuMemcpy2DAsync;
static PFN_cuMemcpy3D_v3020 pfn_cuMemcpy3D;
static PFN_cuMemcpy3DAsync_v3020 pfn_cuMemcpy3DAsync;
static PFN_cuTexObjectCreate_v5000 pfn_cuTexObjectCreate;
static PFN_cuTexObjectDestroy_v5000 pfn_cuTexObjectDestroy;
static PFN_cuMipmappedArrayCreate_v5000 pfn_cuMipmappedArrayCreate;
static PFN_cuMipmappedArrayDestroy_v5000 pfn_cuMipmappedArrayDestroy;
static PFN_cuMipmappedArrayGetLevel_v5000 pfn_cuMipmappedArrayGetLevel;
#endif  // !defined(__HIP_PLATFORM_AMD__)

static bool cuda_driver_initialized = false;

bool ContextGuard::always_restore = false;

CudaTimingState* g_cuda_timing_state = NULL;


static inline int get_major(int version) { return version / 1000; }

static inline int get_minor(int version) { return (version % 1000) / 10; }

// Get versioned driver entry point. The version argument should match the function pointer type.
// For example, to initialize PFN_cuCtxCreate_v3020 use version 3020.
static bool get_driver_entry_point(const char* name, int version, void** pfn)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!name || !pfn)
        return false;
    hipDriverProcAddressQueryResult status = static_cast<hipDriverProcAddressQueryResult>(0);
    const int proc_version = WP_DRIVER_ENTRY_VERSION(version);
    hipError_t r = hipGetProcAddress(name, pfn, proc_version, 0, &status);
    if (r != hipSuccess) {
        fprintf(stderr, "Warp CUDA error: Failed to get HIP entry point '%s' (HIP error %u)\n", name, unsigned(r));
        return false;
    }
    return true;
#else
    if (!pfn_cuGetProcAddress || !name || !pfn)
        return false;

#if CUDA_VERSION < 12000
    CUresult r = pfn_cuGetProcAddress(name, pfn, version, CU_GET_PROC_ADDRESS_DEFAULT);
#else
    CUresult r = pfn_cuGetProcAddress(name, pfn, version, CU_GET_PROC_ADDRESS_DEFAULT, NULL);
#endif

    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "Warp CUDA error: Failed to get driver entry point '%s' (CUDA error %u)\n", name, unsigned(r));
        return false;
    }

    return true;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

bool init_cuda_driver()
{
#if defined(__HIP_PLATFORM_AMD__)
    hipError_t res = hipInit(0);
    if (res != hipSuccess) {
        fprintf(stderr, "Warp CUDA warning: Failed to initialize HIP runtime\n");
        return false;
    }
    cuda_driver_initialized = true;
    return true;
#else
#if defined(_WIN32)
    static HMODULE hCudaDriver = LoadLibraryA("nvcuda.dll");
    if (hCudaDriver == NULL) {
        fprintf(
            stderr,
            "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver. GPU execution will not be available.\n"
        );
        return false;
    }
    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)GetProcAddress(hCudaDriver, "cuGetProcAddress");
#elif defined(__linux__)
    static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    if (hCudaDriver == NULL) {
        // WSL and possibly other systems might require the .1 suffix
        hCudaDriver = dlopen("libcuda.so.1", RTLD_NOW);
        if (hCudaDriver == NULL) {
            fprintf(
                stderr,
                "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver. GPU execution will not be "
                "available.\n"
            );
            return false;
        }
    }
    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(hCudaDriver, "cuGetProcAddress");
#endif

    if (!pfn_cuGetProcAddress) {
        fprintf(stderr, "Warp CUDA error: Failed to get function cuGetProcAddress\n");
        return false;
    }

    // check the CUDA driver version and report an error if it's too low
    int driver_version = 0;
    if (get_driver_entry_point("cuDriverGetVersion", 2020, &(void*&)pfn_cuDriverGetVersion)
        && check_cu(pfn_cuDriverGetVersion(&driver_version))) {
        if (driver_version < WP_CUDA_DRIVER_VERSION) {
            fprintf(
                stderr,
                "Warp CUDA error: Warp requires CUDA driver %d.%d or higher, but the current driver only supports CUDA "
                "%d.%d\n",
                get_major(WP_CUDA_DRIVER_VERSION), get_minor(WP_CUDA_DRIVER_VERSION), get_major(driver_version),
                get_minor(driver_version)
            );
            return false;
        }
    } else {
        fprintf(stderr, "Warp CUDA warning: Unable to determine CUDA driver version\n");
    }

    // initialize driver entry points
    get_driver_entry_point("cuGetErrorString", 6000, &(void*&)pfn_cuGetErrorString);
    get_driver_entry_point("cuGetErrorName", 6000, &(void*&)pfn_cuGetErrorName);
    get_driver_entry_point("cuInit", 2000, &(void*&)pfn_cuInit);
    get_driver_entry_point("cuDeviceGet", 2000, &(void*&)pfn_cuDeviceGet);
    get_driver_entry_point("cuDeviceGetCount", 2000, &(void*&)pfn_cuDeviceGetCount);
    get_driver_entry_point("cuDeviceGetName", 2000, &(void*&)pfn_cuDeviceGetName);
    get_driver_entry_point("cuDeviceGetAttribute", 2000, &(void*&)pfn_cuDeviceGetAttribute);
    get_driver_entry_point("cuDeviceGetUuid", 11040, &(void*&)pfn_cuDeviceGetUuid);
    get_driver_entry_point("cuDevicePrimaryCtxRetain", 7000, &(void*&)pfn_cuDevicePrimaryCtxRetain);
    get_driver_entry_point("cuDevicePrimaryCtxRelease", 11000, &(void*&)pfn_cuDevicePrimaryCtxRelease);
    get_driver_entry_point("cuDeviceCanAccessPeer", 4000, &(void*&)pfn_cuDeviceCanAccessPeer);
    get_driver_entry_point("cuMemGetInfo", 3020, &(void*&)pfn_cuMemGetInfo);
#if CUDA_VERSION >= 12080
    if (driver_version >= 12080)
        get_driver_entry_point("cuMemcpyBatchAsync", 12080, &(void*&)pfn_cuMemcpyBatchAsync);
#endif
    get_driver_entry_point("cuCtxSetCurrent", 4000, &(void*&)pfn_cuCtxSetCurrent);
    get_driver_entry_point("cuCtxGetCurrent", 4000, &(void*&)pfn_cuCtxGetCurrent);
    get_driver_entry_point("cuCtxPushCurrent", 4000, &(void*&)pfn_cuCtxPushCurrent);
    get_driver_entry_point("cuCtxPopCurrent", 4000, &(void*&)pfn_cuCtxPopCurrent);
    get_driver_entry_point("cuCtxSynchronize", 2000, &(void*&)pfn_cuCtxSynchronize);
    get_driver_entry_point("cuCtxGetDevice", 2000, &(void*&)pfn_cuCtxGetDevice);
    get_driver_entry_point("cuCtxCreate", 3020, &(void*&)pfn_cuCtxCreate);
    get_driver_entry_point("cuCtxDestroy", 4000, &(void*&)pfn_cuCtxDestroy);
    get_driver_entry_point("cuCtxEnablePeerAccess", 4000, &(void*&)pfn_cuCtxEnablePeerAccess);
    get_driver_entry_point("cuCtxDisablePeerAccess", 4000, &(void*&)pfn_cuCtxDisablePeerAccess);
    get_driver_entry_point("cuStreamCreate", 2000, &(void*&)pfn_cuStreamCreate);
    get_driver_entry_point("cuStreamDestroy", 4000, &(void*&)pfn_cuStreamDestroy);
    get_driver_entry_point("cuStreamQuery", 2000, &(void*&)pfn_cuStreamQuery);
    get_driver_entry_point("cuStreamSynchronize", 2000, &(void*&)pfn_cuStreamSynchronize);
    get_driver_entry_point("cuStreamWaitEvent", 3020, &(void*&)pfn_cuStreamWaitEvent);
    get_driver_entry_point("cuStreamGetCtx", 9020, &(void*&)pfn_cuStreamGetCtx);
    get_driver_entry_point("cuStreamGetCaptureInfo", 11030, &(void*&)pfn_cuStreamGetCaptureInfo);
    get_driver_entry_point("cuStreamUpdateCaptureDependencies", 11030, &(void*&)pfn_cuStreamUpdateCaptureDependencies);
    get_driver_entry_point("cuStreamCreateWithPriority", 5050, &(void*&)pfn_cuStreamCreateWithPriority);
    get_driver_entry_point("cuStreamGetPriority", 5050, &(void*&)pfn_cuStreamGetPriority);
    get_driver_entry_point("cuEventCreate", 2000, &(void*&)pfn_cuEventCreate);
    get_driver_entry_point("cuEventDestroy", 4000, &(void*&)pfn_cuEventDestroy);
    get_driver_entry_point("cuEventQuery", 2000, &(void*&)pfn_cuEventQuery);
    get_driver_entry_point("cuEventRecord", 2000, &(void*&)pfn_cuEventRecord);
    get_driver_entry_point("cuEventRecordWithFlags", 11010, &(void*&)pfn_cuEventRecordWithFlags);
    get_driver_entry_point("cuEventSynchronize", 2000, &(void*&)pfn_cuEventSynchronize);
#if CUDA_VERSION >= 12030
    if (driver_version >= 12030)
        get_driver_entry_point("cuGraphAddNode", 12030, &(void*&)pfn_cuGraphAddNode);
#endif
    get_driver_entry_point("cuGraphNodeGetDependentNodes", 10000, &(void*&)pfn_cuGraphNodeGetDependentNodes);
    get_driver_entry_point("cuGraphNodeGetType", 10000, &(void*&)pfn_cuGraphNodeGetType);
    get_driver_entry_point("cuModuleLoadDataEx", 2010, &(void*&)pfn_cuModuleLoadDataEx);
    get_driver_entry_point("cuModuleUnload", 2000, &(void*&)pfn_cuModuleUnload);
    get_driver_entry_point("cuModuleGetFunction", 2000, &(void*&)pfn_cuModuleGetFunction);
    get_driver_entry_point("cuLaunchKernel", 4000, &(void*&)pfn_cuLaunchKernel);
    get_driver_entry_point("cuOccupancyMaxPotentialBlockSize", 6050, &(void*&)pfn_cuOccupancyMaxPotentialBlockSize);
    get_driver_entry_point("cuOccupancyMaxActiveClusters", 11070, &(void*&)pfn_cuOccupancyMaxActiveClusters);
    get_driver_entry_point("cuMemcpyPeerAsync", 4000, &(void*&)pfn_cuMemcpyPeerAsync);
    get_driver_entry_point("cuPointerGetAttribute", 4000, &(void*&)pfn_cuPointerGetAttribute);
    get_driver_entry_point("cuGraphicsMapResources", 3000, &(void*&)pfn_cuGraphicsMapResources);
    get_driver_entry_point("cuGraphicsUnmapResources", 3000, &(void*&)pfn_cuGraphicsUnmapResources);
    get_driver_entry_point("cuGraphicsResourceGetMappedPointer", 3020, &(void*&)pfn_cuGraphicsResourceGetMappedPointer);
    get_driver_entry_point("cuGraphicsGLRegisterBuffer", 3000, &(void*&)pfn_cuGraphicsGLRegisterBuffer);
    get_driver_entry_point("cuGraphicsGLRegisterImage", 3000, &(void*&)pfn_cuGraphicsGLRegisterImage);
    get_driver_entry_point(
        "cuGraphicsSubResourceGetMappedArray", 3000, &(void*&)pfn_cuGraphicsSubResourceGetMappedArray
    );
    get_driver_entry_point("cuGraphicsUnregisterResource", 3000, &(void*&)pfn_cuGraphicsUnregisterResource);
    get_driver_entry_point("cuModuleGetGlobal", 3020, &(void*&)pfn_cuModuleGetGlobal);
    get_driver_entry_point("cuFuncSetAttribute", 9000, &(void*&)pfn_cuFuncSetAttribute);
    get_driver_entry_point("cuFuncGetAttribute", 2020, &(void*&)pfn_cuFuncGetAttribute);
    get_driver_entry_point("cuIpcGetEventHandle", 4010, &(void*&)pfn_cuIpcGetEventHandle);
    get_driver_entry_point("cuIpcOpenEventHandle", 4010, &(void*&)pfn_cuIpcOpenEventHandle);
    get_driver_entry_point("cuIpcGetMemHandle", 4010, &(void*&)pfn_cuIpcGetMemHandle);
    get_driver_entry_point("cuIpcOpenMemHandle", 11000, &(void*&)pfn_cuIpcOpenMemHandle);
    get_driver_entry_point("cuIpcCloseMemHandle", 4010, &(void*&)pfn_cuIpcCloseMemHandle);

    // Profiler control functions
    get_driver_entry_point("cuProfilerStart", 4000, &(void*&)pfn_cuProfilerStart);
    get_driver_entry_point("cuProfilerStop", 4000, &(void*&)pfn_cuProfilerStop);

    // Texture functions
    get_driver_entry_point("cuArrayCreate", 3020, &(void*&)pfn_cuArrayCreate);
    get_driver_entry_point("cuArrayDestroy", 2000, &(void*&)pfn_cuArrayDestroy);
    get_driver_entry_point("cuArray3DCreate", 3020, &(void*&)pfn_cuArray3DCreate);
    get_driver_entry_point("cuArray3DGetDescriptor", 3020, &(void*&)pfn_cuArray3DGetDescriptor);
    get_driver_entry_point("cuMemcpy2D", 3020, &(void*&)pfn_cuMemcpy2D);
    get_driver_entry_point("cuMemcpy2DAsync", 3020, &(void*&)pfn_cuMemcpy2DAsync);
    get_driver_entry_point("cuMemcpy3D", 3020, &(void*&)pfn_cuMemcpy3D);
    get_driver_entry_point("cuMemcpy3DAsync", 3020, &(void*&)pfn_cuMemcpy3DAsync);
    get_driver_entry_point("cuTexObjectCreate", 5000, &(void*&)pfn_cuTexObjectCreate);
    get_driver_entry_point("cuTexObjectDestroy", 5000, &(void*&)pfn_cuTexObjectDestroy);
    get_driver_entry_point("cuMipmappedArrayCreate", 5000, &(void*&)pfn_cuMipmappedArrayCreate);
    get_driver_entry_point("cuMipmappedArrayDestroy", 5000, &(void*&)pfn_cuMipmappedArrayDestroy);
    get_driver_entry_point("cuMipmappedArrayGetLevel", 5000, &(void*&)pfn_cuMipmappedArrayGetLevel);

    if (pfn_cuInit)
        cuda_driver_initialized = check_cu(pfn_cuInit(0));

    return cuda_driver_initialized;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

bool is_cuda_driver_initialized() { return cuda_driver_initialized; }

bool check_cuda_result(cudaError_t code, const char* func, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    wp::set_error_string(
        "Warp CUDA error %u: %s (in function %s, %s:%d)", unsigned(code), cudaGetErrorString(code), func, file, line
    );
    return false;
}

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    if (result == CUDA_SUCCESS)
        return true;

#if defined(__HIP_PLATFORM_AMD__)
    const char* errString = hipGetErrorString(result);
#else
    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);
#endif  // defined(__HIP_PLATFORM_AMD__)

    if (errString)
        wp::set_error_string(
            "Warp CUDA error %u: %s (in function %s, %s:%d)", unsigned(result), errString, func, file, line
        );
    else
        wp::set_error_string("Warp CUDA error %u (in function %s, %s:%d)", unsigned(result), func, file, line);

    return false;
}

bool get_capture_dependencies(CUstream stream, std::vector<CUgraphNode>& dependencies_ret)
{
    CUstreamCaptureStatus status;
    size_t num_dependencies = 0;
    const CUgraphNode* dependencies = NULL;
    dependencies_ret.clear();
    if (check_cu(cuStreamGetCaptureInfo_f(stream, &status, NULL, NULL, &dependencies, &num_dependencies))) {
        if (dependencies && num_dependencies > 0)
            dependencies_ret.insert(dependencies_ret.begin(), dependencies, dependencies + num_dependencies);
        return true;
    }
    return false;
}

bool get_graph_leaf_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    if (!graph)
        return false;

    size_t node_count = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, NULL, &node_count)))
        return false;

    std::vector<cudaGraphNode_t> nodes(node_count);
    if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &node_count)))
        return false;

    leaf_nodes_ret.clear();

    for (cudaGraphNode_t node : nodes) {
        size_t dependent_count;

        if (!check_cu(cuGraphNodeGetDependentNodes_f(node, NULL, &dependent_count)))
            return false;

        if (dependent_count == 0)
            leaf_nodes_ret.push_back(node);
    }

    return true;
}

// get all leaf nodes that depend on the given ancestor node
bool get_dependent_leaf_nodes(cudaGraphNode_t ancestor, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    if (!ancestor)
        return false;

    std::queue<cudaGraphNode_t> frontier { { ancestor } };
    std::unordered_set<cudaGraphNode_t> visited { ancestor };
    std::vector<cudaGraphNode_t> deps;

    leaf_nodes_ret.clear();

    while (!frontier.empty()) {
        cudaGraphNode_t node = frontier.front();
        frontier.pop();

        size_t dep_count = 0;
        if (!check_cu(cuGraphNodeGetDependentNodes_f(node, NULL, &dep_count)))
            return false;

        if (dep_count == 0) {
            leaf_nodes_ret.push_back(node);
        } else {
            deps.resize(dep_count);
            if (!check_cu(cuGraphNodeGetDependentNodes_f(node, deps.data(), &dep_count)))
                return false;
            for (cudaGraphNode_t dep : deps) {
                if (visited.insert(dep).second) {
                    frontier.push(dep);
                }
            }
        }
    }

    return true;
}

// whether argument node depends on referent node
NodeDependencyResult graph_node_depends_on(cudaGraphNode_t argument, cudaGraphNode_t referent)
{
    if (!argument || !referent)
        return NODE_DEPENDENCY_RESULT_ERROR;

    std::queue<cudaGraphNode_t> frontier { { referent } };
    std::unordered_set<cudaGraphNode_t> visited { referent };
    std::vector<cudaGraphNode_t> deps;

    while (!frontier.empty()) {
        cudaGraphNode_t node = frontier.front();
        frontier.pop();

        if (node == argument)
            return NODE_DEPENDENCY_RESULT_DEPENDENT;

        size_t dep_count = 0;
        if (!check_cu(cuGraphNodeGetDependentNodes_f(node, NULL, &dep_count)))
            return NODE_DEPENDENCY_RESULT_ERROR;

        if (dep_count > 0) {
            deps.resize(dep_count);
            if (!check_cu(cuGraphNodeGetDependentNodes_f(node, deps.data(), &dep_count)))
                return NODE_DEPENDENCY_RESULT_ERROR;
            for (cudaGraphNode_t dep : deps) {
                if (visited.insert(dep).second) {
                    frontier.push(dep);
                }
            }
        }
    }

    return NODE_DEPENDENCY_RESULT_INDEPENDENT;
}

// determine the status of an allocation at the given query node
GraphAllocQueryResult graph_alloc_query(cudaGraphNode_t alloc_node, cudaGraphNode_t query_node)
{
    if (!alloc_node || !query_node)
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;

    CUgraphNodeType alloc_node_type;
    if (!check_cu(cuGraphNodeGetType_f(alloc_node, &alloc_node_type)))
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;
    if (alloc_node_type != CU_GRAPH_NODE_TYPE_MEM_ALLOC)
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;

    // get the allocation pointer so we can locate the matching free node (if any)
    cudaMemAllocNodeParams alloc_params;
    if (!check_cuda(cudaGraphMemAllocNodeGetParams(alloc_node, &alloc_params)))
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;
    void* alloc_ptr = alloc_params.dptr;

    // BFS from the alloc node to locate the matching free node and to
    // determine whether the query node is a descendant of the alloc.
    std::queue<cudaGraphNode_t> frontier { { alloc_node } };
    std::unordered_set<cudaGraphNode_t> visited { alloc_node };
    std::vector<cudaGraphNode_t> deps;
    cudaGraphNode_t free_node = NULL;
    bool query_reachable = false;

    while (!frontier.empty()) {
        cudaGraphNode_t node = frontier.front();
        frontier.pop();

        // record if the query node is reachable from the alloc
        if (node == query_node)
            query_reachable = true;

        // record the first free node that matches the alloc pointer
        // - there should only be one free for this alloc, we're not tackling
        //   double-free errors here.
        // - after the alloc is freed, subsequent alloc and free nodes can reuse
        //   the same pointer value, but we don't care about those.
        if (!free_node && node != alloc_node) {
            CUgraphNodeType node_type;
            if (!check_cu(cuGraphNodeGetType_f(node, &node_type)))
                return GRAPH_ALLOC_QUERY_RESULT_ERROR;
            if (node_type == CU_GRAPH_NODE_TYPE_MEM_FREE) {
                void* free_ptr = NULL;
                if (!check_cuda(cudaGraphMemFreeNodeGetParams(node, &free_ptr)))
                    return GRAPH_ALLOC_QUERY_RESULT_ERROR;
                if (free_ptr == alloc_ptr)
                    free_node = node;
            }
        }

        size_t dep_count = 0;
        if (!check_cu(cuGraphNodeGetDependentNodes_f(node, NULL, &dep_count)))
            return GRAPH_ALLOC_QUERY_RESULT_ERROR;

        if (dep_count > 0) {
            deps.resize(dep_count);
            if (!check_cu(cuGraphNodeGetDependentNodes_f(node, deps.data(), &dep_count)))
                return GRAPH_ALLOC_QUERY_RESULT_ERROR;
            for (cudaGraphNode_t dep : deps) {
                if (visited.insert(dep).second)
                    frontier.push(dep);
            }
        }
    }

    if (!query_reachable) {
        // query node does not depend on alloc, so allocation is inaccessible
        return GRAPH_ALLOC_QUERY_RESULT_INACCESSIBLE;
    }

    if (!free_node) {
        // alloc is never freed in the graph
        return GRAPH_ALLOC_QUERY_RESULT_AVAILABLE;
    }

    // Query node depends on the alloc and a free node was found, so we need to check
    // the relationship between the query node and the free node:
    // - If the query node depends on the free node, then the allocation is guaranteed
    //   to be freed before the query node is reached.
    // - If the free node depends on the query node, then the allocation is guaranteed
    //   to be available when the query node is reached.
    // - If the query node and free node are independent, then they can execute
    //   concurrently leading to potential use-after-free errors.

    // check if the query node executes after the free node
    NodeDependencyResult q_after_f = graph_node_depends_on(query_node, free_node);
    if (q_after_f == NODE_DEPENDENCY_RESULT_ERROR)
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;
    if (q_after_f == NODE_DEPENDENCY_RESULT_DEPENDENT)
        return GRAPH_ALLOC_QUERY_RESULT_FREED;

    // check if the free node executes after the query node
    NodeDependencyResult f_after_q = graph_node_depends_on(free_node, query_node);
    if (f_after_q == NODE_DEPENDENCY_RESULT_ERROR)
        return GRAPH_ALLOC_QUERY_RESULT_ERROR;
    if (f_after_q == NODE_DEPENDENCY_RESULT_DEPENDENT)
        return GRAPH_ALLOC_QUERY_RESULT_AVAILABLE;

    // free is independent of the query node, potentially causing use-after-free errors
    return GRAPH_ALLOC_QUERY_RESULT_USE_AFTER_FREE;
}

#define DRIVER_ENTRY_POINT_ERROR driver_entry_point_error(__FUNCTION__)

static CUresult driver_entry_point_error(const char* function)
{
    fprintf(stderr, "Warp CUDA error: Function %s: a suitable driver entry point was not found\n", function);
#if defined(__HIP_PLATFORM_AMD__)
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return (CUresult)cudaErrorCallRequiresNewerDriver;  // this matches what cudart would do
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDriverGetVersion_f(int* version)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDriverGetVersion(version);
#else
    return pfn_cuDriverGetVersion ? pfn_cuDriverGetVersion(version) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
#if defined(__HIP_PLATFORM_AMD__)
    const char* err = hipGetErrorName(result);
    if (pstr)
        *pstr = err;
    return err ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuGetErrorName ? pfn_cuGetErrorName(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
#if defined(__HIP_PLATFORM_AMD__)
    const char* err = hipGetErrorString(result);
    if (pstr)
        *pstr = err;
    return err ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuGetErrorString ? pfn_cuGetErrorString(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuInit_f(unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipInit(flags);
#else
    return pfn_cuInit ? pfn_cuInit(flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceGet_f(CUdevice* dev, int ordinal)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDeviceGet(dev, ordinal);
#else
    return pfn_cuDeviceGet ? pfn_cuDeviceGet(dev, ordinal) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceGetCount_f(int* count)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!count)
        return CUDA_ERROR_NOT_INITIALIZED;
    return hipGetDeviceCount(count);
#else
    if (pfn_cuDeviceGetCount)
        return pfn_cuDeviceGetCount(count);

    // allow calling this function even if CUDA is not available
    if (count)
        *count = 0;

    return CUDA_SUCCESS;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDeviceGetName(name, len, dev);
#else
    return pfn_cuDeviceGetName ? pfn_cuDeviceGetName(name, len, dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (attrib == CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED) {
        if (value)
            *value = 0;
        return CUDA_SUCCESS;
    }
    return hipDeviceGetAttribute(value, static_cast<hipDeviceAttribute_t>(attrib), dev);
#else
    return pfn_cuDeviceGetAttribute ? pfn_cuDeviceGetAttribute(value, attrib, dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!uuid)
        return CUDA_ERROR_NOT_SUPPORTED;
    return hipDeviceGetUuid(uuid, dev);
#else
    return pfn_cuDeviceGetUuid ? pfn_cuDeviceGetUuid(uuid, dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    HipContext* context = get_hip_context_for_device(static_cast<int>(dev));
    if (!context)
        return CUDA_ERROR_NOT_INITIALIZED;
    *ctx = context;
    return CUDA_SUCCESS;
#else
    return pfn_cuDevicePrimaryCtxRetain ? pfn_cuDevicePrimaryCtxRetain(ctx, dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)dev;
    return CUDA_SUCCESS;
#else
    return pfn_cuDevicePrimaryCtxRelease ? pfn_cuDevicePrimaryCtxRelease(dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDeviceCanAccessPeer(can_access, dev, peer_dev);
#else
    return pfn_cuDeviceCanAccessPeer ? pfn_cuDeviceCanAccessPeer(can_access, dev, peer_dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemGetInfo_f(size_t* free, size_t* total)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipMemGetInfo(free, total);
#else
    return pfn_cuMemGetInfo ? pfn_cuMemGetInfo(free, total) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

#if WP_HAS_MEMCPY_BATCH
CUresult cuMemcpyBatchAsync_f(
    CUdeviceptr* dsts,
    CUdeviceptr* srcs,
    size_t* sizes,
    size_t count,
    CUmemcpyAttributes* attrs,
    size_t* attrsIdxs,
    size_t numAttrs,
    size_t* failIdx,
    CUstream hStream
)
{
#if defined(__HIP_PLATFORM_AMD__)
#if HIP_VERSION >= 70100000
    return hipMemcpyBatchAsync(
        reinterpret_cast<void**>(dsts),
        reinterpret_cast<void**>(srcs),
        sizes,
        count,
        attrs,
        attrsIdxs,
        numAttrs,
        failIdx,
        hStream
    );
#else
    (void)dsts;
    (void)srcs;
    (void)sizes;
    (void)count;
    (void)attrs;
    (void)attrsIdxs;
    (void)numAttrs;
    (void)failIdx;
    (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif  // HIP_VERSION >= 70100000
#else
    return pfn_cuMemcpyBatchAsync
        ? pfn_cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, hStream)
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}
#endif  // WP_HAS_MEMCPY_BATCH

CUresult cuCtxGetCurrent_f(CUcontext* ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    int device = -1;
    hipError_t res = hipGetDevice(&device);
    if (res != hipSuccess)
        return res;
    HipContext* context = get_hip_context_for_device(device);
    if (!context)
        return CUDA_ERROR_NOT_INITIALIZED;
    *ctx = context;
    return CUDA_SUCCESS;
#else
    return pfn_cuCtxGetCurrent ? pfn_cuCtxGetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxSetCurrent_f(CUcontext ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!ctx)
        return CUDA_SUCCESS;
    return hipSetDevice(ctx->device);
#else
    return pfn_cuCtxSetCurrent ? pfn_cuCtxSetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxPushCurrent_f(CUcontext ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    CUcontext current = nullptr;
    CUresult result = cuCtxGetCurrent_f(&current);
    if (result != CUDA_SUCCESS)
        return result;
    g_hip_context_stack.push_back(current);
    if (ctx)
        return cuCtxSetCurrent_f(ctx);
    return CUDA_SUCCESS;
#else
    return pfn_cuCtxPushCurrent ? pfn_cuCtxPushCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxPopCurrent_f(CUcontext* ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (g_hip_context_stack.empty())
        return CUDA_ERROR_NOT_INITIALIZED;
    CUcontext current = nullptr;
    CUresult result = cuCtxGetCurrent_f(&current);
    if (result != CUDA_SUCCESS)
        return result;
    CUcontext previous = g_hip_context_stack.back();
    g_hip_context_stack.pop_back();
    *ctx = current;
    return cuCtxSetCurrent_f(previous);
#else
    return pfn_cuCtxPopCurrent ? pfn_cuCtxPopCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxSynchronize_f()
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDeviceSynchronize();
#else
    return pfn_cuCtxSynchronize ? pfn_cuCtxSynchronize() : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuProfilerStart_f()
{
#if defined(__HIP_PLATFORM_AMD__)
    // hipProfilerStart is deprecated (use roctracer/rocTX); treat as a no-op.
    return CUDA_SUCCESS;
#else
    return pfn_cuProfilerStart ? pfn_cuProfilerStart() : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuProfilerStop_f()
{
#if defined(__HIP_PLATFORM_AMD__)
    // hipProfilerStop is deprecated (use roctracer/rocTX); treat as a no-op.
    return CUDA_SUCCESS;
#else
    return pfn_cuProfilerStop ? pfn_cuProfilerStop() : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxGetDevice_f(CUdevice* dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!dev)
        return CUDA_ERROR_NOT_INITIALIZED;
    int device = -1;
    hipError_t res = hipGetDevice(&device);
    if (res != hipSuccess)
        return res;
    *dev = static_cast<CUdevice>(device);
    return CUDA_SUCCESS;
#else
    return pfn_cuCtxGetDevice ? pfn_cuCtxGetDevice(dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)flags;
    if (!ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    HipContext* context = get_hip_context_for_device(static_cast<int>(dev));
    if (!context)
        return CUDA_ERROR_NOT_INITIALIZED;
    *ctx = context;
    return hipSetDevice(context->device);
#else
    return pfn_cuCtxCreate ? pfn_cuCtxCreate(ctx, flags, dev) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxDestroy_f(CUcontext ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)ctx;
    return CUDA_SUCCESS;
#else
    return pfn_cuCtxDestroy ? pfn_cuCtxDestroy(ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!peer_ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    return hipDeviceEnablePeerAccess(peer_ctx->device, flags);
#else
    return pfn_cuCtxEnablePeerAccess ? pfn_cuCtxEnablePeerAccess(peer_ctx, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!peer_ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    return hipDeviceDisablePeerAccess(peer_ctx->device);
#else
    return pfn_cuCtxDisablePeerAccess ? pfn_cuCtxDisablePeerAccess(peer_ctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamCreateWithFlags(stream, flags);
#else
    return pfn_cuStreamCreate ? pfn_cuStreamCreate(stream, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamDestroy_f(CUstream stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamDestroy(stream);
#else
    return pfn_cuStreamDestroy ? pfn_cuStreamDestroy(stream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamQuery_f(CUstream stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamQuery(stream);
#else
    return pfn_cuStreamQuery ? pfn_cuStreamQuery(stream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamSynchronize_f(CUstream stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamSynchronize(stream);
#else
    return pfn_cuStreamSynchronize ? pfn_cuStreamSynchronize(stream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamWaitEvent(stream, event, flags);
#else
    return pfn_cuStreamWaitEvent ? pfn_cuStreamWaitEvent(stream, event, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)stream;
    (void)pctx;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuStreamGetCtx ? pfn_cuStreamGetCtx(stream, pctx) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamGetCaptureInfo_f(
    CUstream stream,
    CUstreamCaptureStatus* captureStatus_out,
    cuuint64_t* id_out,
    CUgraph* graph_out,
    const CUgraphNode** dependencies_out,
    size_t* numDependencies_out
)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamGetCaptureInfo_v2(stream, captureStatus_out,
        reinterpret_cast<unsigned long long*>(id_out), graph_out, dependencies_out, numDependencies_out);
#else
    return pfn_cuStreamGetCaptureInfo
        ? pfn_cuStreamGetCaptureInfo(
              stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out
          )
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamUpdateCaptureDependencies_f(
    CUstream stream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags
)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
#else
    return pfn_cuStreamUpdateCaptureDependencies
        ? pfn_cuStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamCreateWithPriority_f(CUstream* phStream, unsigned int flags, int priority)
{
#if defined(__HIP_PLATFORM_AMD__)
     return hipStreamCreateWithPriority(phStream, flags, priority);
#else
    return pfn_cuStreamCreateWithPriority ? pfn_cuStreamCreateWithPriority(phStream, flags, priority)
                                          : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuStreamGetPriority_f(CUstream hStream, int* priority)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipStreamGetPriority(hStream, priority);
#else
    return pfn_cuStreamGetPriority ? pfn_cuStreamGetPriority(hStream, priority) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventCreate_f(CUevent* event, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipEventCreateWithFlags(event, flags);
#else
    return pfn_cuEventCreate ? pfn_cuEventCreate(event, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventDestroy_f(CUevent event)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipEventDestroy(event);
#else
    return pfn_cuEventDestroy ? pfn_cuEventDestroy(event) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventQuery_f(CUevent event)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipEventQuery(event);
#else
    return pfn_cuEventQuery ? pfn_cuEventQuery(event) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventRecord_f(CUevent event, CUstream stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipEventRecord(event, stream);
#else
    return pfn_cuEventRecord ? pfn_cuEventRecord(event, stream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)flags;
    return hipEventRecord(event, stream);
#else
    return pfn_cuEventRecordWithFlags ? pfn_cuEventRecordWithFlags(event, stream, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuEventSynchronize_f(CUevent event)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipEventSynchronize(event);
#else
    return pfn_cuEventSynchronize ? pfn_cuEventSynchronize(event) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION >= 12030
CUresult cuGraphAddNode_f(
    CUgraphNode* phGraphNode,
    CUgraph hGraph,
    const CUgraphNode* dependencies,
    const CUgraphEdgeData* dependencyData,
    size_t numDependencies,
    CUgraphNodeParams* nodeParams
)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)phGraphNode;
    (void)hGraph;
    (void)dependencies;
    (void)dependencyData;
    (void)numDependencies;
    (void)nodeParams;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuGraphAddNode
        ? pfn_cuGraphAddNode(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}
#endif  // defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION >= 12030

CUresult cuGraphNodeGetDependentNodes_f(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
#else
    return pfn_cuGraphNodeGetDependentNodes ? pfn_cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
                                            : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphNodeGetType_f(CUgraphNode hNode, CUgraphNodeType* type)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphNodeGetType(hNode, type);
#else
    return pfn_cuGraphNodeGetType ? pfn_cuGraphNodeGetType(hNode, type) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuModuleLoadDataEx_f(
    CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues
)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return hipModuleLoadData(module, image);
#else
    return pfn_cuModuleLoadDataEx ? pfn_cuModuleLoadDataEx(module, image, numOptions, options, optionValues)
                                  : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuModuleUnload_f(CUmodule hmod)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipModuleUnload(hmod);
#else
    return pfn_cuModuleUnload ? pfn_cuModuleUnload(hmod) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuModuleGetFunction_f(CUfunction* hfunc, CUmodule hmod, const char* name)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipModuleGetFunction(hfunc, hmod, name);
#else
    return pfn_cuModuleGetFunction ? pfn_cuModuleGetFunction(hfunc, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuLaunchKernel_f(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipModuleLaunchKernel(
        f,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        hStream,
        kernelParams,
        extra
    );
#else
    return pfn_cuLaunchKernel ? pfn_cuLaunchKernel(
                                    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                                    hStream, kernelParams, extra
                                )
                              : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuOccupancyMaxPotentialBlockSize_f(
    int* minGridSize,
    int* blockSize,
    CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize,
    int blockSizeLimit
)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP has no equivalent that takes a B2DSize callback. The only Warp caller
    // passes NULL for blockSizeToDynamicSMemSize, so silently ignore it and use
    // the fixed-shared-memory variant.
    (void)blockSizeToDynamicSMemSize;
    return hipModuleOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, dynamicSMemSize, blockSizeLimit);
#else
    return pfn_cuOccupancyMaxPotentialBlockSize
        ? pfn_cuOccupancyMaxPotentialBlockSize(
              minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit
          )
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemcpyPeerAsync_f(
    CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream
)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (!dst_ctx || !src_ctx)
        return CUDA_ERROR_NOT_INITIALIZED;
    return hipMemcpyPeerAsync(
        reinterpret_cast<void*>(dst_ptr),
        dst_ctx->device,
        reinterpret_cast<const void*>(src_ptr),
        src_ctx->device,
        n,
        stream
    );
#else
    return pfn_cuMemcpyPeerAsync ? pfn_cuMemcpyPeerAsync(dst_ptr, dst_ctx, src_ptr, src_ctx, n, stream)
                                 : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipPointerGetAttribute(data, attribute, ptr);
#else
    return pfn_cuPointerGetAttribute ? pfn_cuPointerGetAttribute(data, attribute, ptr) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphicsMapResources(count, resources, stream);
#else
    return pfn_cuGraphicsMapResources ? pfn_cuGraphicsMapResources(count, resources, stream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphicsUnmapResources(count, resources, hStream);
#else
    return pfn_cuGraphicsUnmapResources ? pfn_cuGraphicsUnmapResources(count, resources, hStream)
                                        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(pDevPtr), pSize, resource);
#else
    return pfn_cuGraphicsResourceGetMappedPointer ? pfn_cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource)
                                                  : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource* pCudaResource, unsigned int buffer, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)pCudaResource;
    (void)buffer;
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuGraphicsGLRegisterBuffer ? pfn_cuGraphicsGLRegisterBuffer(pCudaResource, (wp::GLuint)buffer, flags)
                                          : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsGLRegisterImage_f(
    CUgraphicsResource* pCudaResource, unsigned int image, unsigned int target, unsigned int flags
)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)pCudaResource;
    (void)image;
    (void)target;
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuGraphicsGLRegisterImage ? pfn_cuGraphicsGLRegisterImage(pCudaResource, image, target, flags)
                                         : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsSubResourceGetMappedArray_f(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel
)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
#else
    return pfn_cuGraphicsSubResourceGetMappedArray
        ? pfn_cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipGraphicsUnregisterResource(resource);
#else
    return pfn_cuGraphicsUnregisterResource ? pfn_cuGraphicsUnregisterResource(resource) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(dptr), bytes, hmod, name);
#else
    return pfn_cuModuleGetGlobal ? pfn_cuModuleGetGlobal(dptr, bytes, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuOccupancyMaxActiveClusters_f(int* numClusters, CUfunction func, const CUlaunchConfig* config)
{
#if defined(__HIP_PLATFORM_AMD__)
    // Thread-block clusters are a CUDA-only feature; HIP/ROCm has no equivalent.
    (void)numClusters;
    (void)func;
    (void)config;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return pfn_cuOccupancyMaxActiveClusters ? pfn_cuOccupancyMaxActiveClusters(numClusters, func, config)
                                            : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP has no driver-style setter for module functions. The attributes Warp
    // sets here are either dynamic-shared-memory sizing (specified at launch
    // time on HIP) or non-portable cluster opt-in (clusters are unsupported on
    // HIP), so treat this as a successful no-op.
    (void)hfunc; (void)attrib; (void)value;
    return hipSuccess;
#else
    return pfn_cuFuncSetAttribute ? pfn_cuFuncSetAttribute(hfunc, attrib, value) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuFuncGetAttribute_f(int* pi, CUfunction_attribute attrib, CUfunction hfunc)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipFuncGetAttribute(pi, attrib, hfunc);
#else
    return pfn_cuFuncGetAttribute ? pfn_cuFuncGetAttribute(pi, attrib, hfunc) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuIpcGetEventHandle_f(CUipcEventHandle* pHandle, CUevent event)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipIpcGetEventHandle(pHandle, event);
#else
    return pfn_cuIpcGetEventHandle ? pfn_cuIpcGetEventHandle(pHandle, event) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuIpcOpenEventHandle_f(CUevent* phEvent, CUipcEventHandle handle)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipIpcOpenEventHandle(phEvent, handle);
#else
    return pfn_cuIpcOpenEventHandle ? pfn_cuIpcOpenEventHandle(phEvent, handle) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuIpcGetMemHandle_f(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
 #if defined(__HIP_PLATFORM_AMD__)
    return hipIpcGetMemHandle(pHandle, reinterpret_cast<void*>(dptr));
#else
    return pfn_cuIpcGetMemHandle ? pfn_cuIpcGetMemHandle(pHandle, dptr) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuIpcOpenMemHandle_f(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipIpcOpenMemHandle(reinterpret_cast<void**>(pdptr), handle, flags);
#else
    return pfn_cuIpcOpenMemHandle ? pfn_cuIpcOpenMemHandle(pdptr, handle, flags) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipIpcCloseMemHandle(reinterpret_cast<void*>(dptr));
#else
    return pfn_cuIpcCloseMemHandle ? pfn_cuIpcCloseMemHandle(dptr) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

// Texture functions
CUresult cuArrayCreate_f(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
#if defined(__HIP_PLATFORM_AMD__)
    // On some CDNA targets (notably gfx94x/gfx95x builds), HIP exposes texture APIs but
    // marks image support as unavailable via __HIP_NO_IMAGE_SUPPORT.
#if defined(__HIP_NO_IMAGE_SUPPORT)
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
    return hipArrayCreate(pHandle, pAllocateArray);
#else
    return pfn_cuArrayCreate ? pfn_cuArrayCreate(pHandle, pAllocateArray) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuArrayDestroy_f(CUarray hArray)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipArrayDestroy(hArray);
#else
    return pfn_cuArrayDestroy ? pfn_cuArrayDestroy(hArray) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuArray3DCreate_f(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
#if defined(__HIP_PLATFORM_AMD__)
    // On some CDNA targets (notably gfx94x/gfx95x builds), HIP exposes texture APIs but
    // marks image support as unavailable via __HIP_NO_IMAGE_SUPPORT.
#if defined(__HIP_NO_IMAGE_SUPPORT)
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
    return hipArray3DCreate(pHandle, pAllocateArray);
#else
    return pfn_cuArray3DCreate ? pfn_cuArray3DCreate(pHandle, pAllocateArray) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
#if defined(__HIP_PLATFORM_AMD__)
#if defined(__HIP_NO_IMAGE_SUPPORT)
    (void)pArrayDescriptor;
    (void)hArray;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
    return hipArray3DGetDescriptor(pArrayDescriptor, hArray);
#else
    return pfn_cuArray3DGetDescriptor ? pfn_cuArray3DGetDescriptor(pArrayDescriptor, hArray) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemcpy2D_f(const CUDA_MEMCPY2D* pCopy)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipMemcpyParam2D(pCopy);
#else
    return pfn_cuMemcpy2D ? pfn_cuMemcpy2D(pCopy) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemcpy2DAsync_f(const CUDA_MEMCPY2D* pCopy, CUstream hStream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipMemcpyParam2DAsync(pCopy, hStream);
#else
    return pfn_cuMemcpy2DAsync ? pfn_cuMemcpy2DAsync(pCopy, hStream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemcpy3D_f(const CUDA_MEMCPY3D* pCopy)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDrvMemcpy3D(pCopy);
#else
    return pfn_cuMemcpy3D ? pfn_cuMemcpy3D(pCopy) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipDrvMemcpy3DAsync(pCopy, hStream);
#else
    return pfn_cuMemcpy3DAsync ? pfn_cuMemcpy3DAsync(pCopy, hStream) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuTexObjectCreate_f(
    CUtexObject* pTexObject,
    const CUDA_RESOURCE_DESC* pResDesc,
    const CUDA_TEXTURE_DESC* pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc
)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
#else
    return pfn_cuTexObjectCreate ? pfn_cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
                                 : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuTexObjectDestroy_f(CUtexObject texObject)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipTexObjectDestroy(texObject);
#else
    return pfn_cuTexObjectDestroy ? pfn_cuTexObjectDestroy(texObject) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMipmappedArrayCreate_f(
    CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels
)
{
#if defined(__HIP_PLATFORM_AMD__)
#if defined(__HIP_NO_IMAGE_SUPPORT)
    (void)pHandle;
    (void)pMipmappedArrayDesc;
    (void)numMipmapLevels;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    return hipMipmappedArrayCreate(pHandle, const_cast<CUDA_ARRAY3D_DESCRIPTOR*>(pMipmappedArrayDesc), numMipmapLevels);
#endif
#else
    return pfn_cuMipmappedArrayCreate ? pfn_cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
                                      : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMipmappedArrayDestroy_f(CUmipmappedArray hMipmappedArray)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipMipmappedArrayDestroy(hMipmappedArray);
#else
    return pfn_cuMipmappedArrayDestroy ? pfn_cuMipmappedArrayDestroy(hMipmappedArray) : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

CUresult cuMipmappedArrayGetLevel_f(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
#else
    return pfn_cuMipmappedArrayGetLevel ? pfn_cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)
                                        : DRIVER_ENTRY_POINT_ERROR;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

#endif  // WP_ENABLE_CUDA
