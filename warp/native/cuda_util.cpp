// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if WP_ENABLE_CUDA

#include "cuda_util.h"
#include "error.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wingdi.h>  // needed for OpenGL includes
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <queue>
#include <unordered_set>
#include <vector>

#if WP_ENABLE_HIP

#include <hip/hip_gl_interop.h>

// HIP has no separately loaded driver library: the driver-style entry points are part of the HIP
// runtime that warp.so links. The cu*_f wrappers below call it directly.
//
// HIP also has no driver-API contexts (the hipCtx API is deprecated). A Warp context is a handle
// to a device: each device has a primary context handle, created at initialization, and
// cuCtxCreate_f makes additional handles that share the device.

struct WpHipContext {
    hipDevice_t device;
    bool is_primary;
};

static std::vector<WpHipContext*> g_hip_primary_contexts;

// the context handle made current on this thread, if it is not a primary context
static thread_local CUcontext g_hip_current_context = NULL;
static thread_local std::vector<CUcontext> g_hip_context_stack;

// used by check_cu_result(), which ignores the result
static void get_error_string(hipError_t result, const char** pstr) { (void)hipDrvGetErrorString(result, pstr); }
static void (*pfn_cuGetErrorString)(hipError_t, const char**) = get_error_string;

static bool cuda_driver_initialized = false;

bool ContextGuard::always_restore = false;

CudaTimingState* g_cuda_timing_state = NULL;

static CUcontext get_primary_context(hipDevice_t device)
{
    if (device < 0 || device >= int(g_hip_primary_contexts.size()))
        return NULL;
    return g_hip_primary_contexts[device];
}

bool init_cuda_driver()
{
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        fprintf(
            stderr,
            "Warp CUDA warning: Could not initialize the HIP runtime (%s). GPU execution will not be available.\n",
            hipGetErrorString(result)
        );
        return false;
    }

    int device_count = 0;
    if (!check_cu(hipGetDeviceCount(&device_count)))
        return false;

    for (int i = 0; i < device_count; i++)
        g_hip_primary_contexts.push_back(new WpHipContext { i, true });

    cuda_driver_initialized = true;
    return true;
}

CUresult cuDriverGetVersion_f(int* version) { return hipDriverGetVersion(version); }

CUresult cuGetErrorName_f(CUresult result, const char** pstr) { return hipDrvGetErrorName(result, pstr); }

CUresult cuGetErrorString_f(CUresult result, const char** pstr) { return hipDrvGetErrorString(result, pstr); }

CUresult cuInit_f(unsigned int flags) { return hipInit(flags); }

CUresult cuDeviceGet_f(CUdevice* dev, int ordinal) { return hipDeviceGet(dev, ordinal); }

CUresult cuDeviceGetCount_f(int* count)
{
    if (cuda_driver_initialized)
        return hipGetDeviceCount(count);

    // allow calling this function even if HIP is not available
    if (count)
        *count = 0;

    return hipSuccess;
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev) { return hipDeviceGetName(name, len, dev); }

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return hipDeviceGetAttribute(value, attrib, dev);
}

CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev) { return hipDeviceGetUuid(uuid, dev); }

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    *ctx = get_primary_context(dev);
    return *ctx ? hipSuccess : hipErrorInvalidDevice;
}

CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev)
{
    return get_primary_context(dev) ? hipSuccess : hipErrorInvalidDevice;
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
    return hipDeviceCanAccessPeer(can_access, dev, peer_dev);
}

CUresult cuMemGetInfo_f(size_t* free, size_t* total) { return hipMemGetInfo(free, total); }

CUresult cuCtxGetCurrent_f(CUcontext* ctx)
{
    hipDevice_t device;
    hipError_t result = hipGetDevice(&device);
    if (result != hipSuccess)
        return result;

    // the HIP runtime always has a current device, which other HIP code (e.g. PyTorch) may have set
    if (g_hip_current_context && g_hip_current_context->device == device)
        *ctx = g_hip_current_context;
    else
        *ctx = get_primary_context(device);

    return hipSuccess;
}

CUresult cuCtxSetCurrent_f(CUcontext ctx)
{
    if (ctx) {
        hipError_t result = hipSetDevice(ctx->device);
        if (result != hipSuccess)
            return result;
    }
    g_hip_current_context = (ctx && !ctx->is_primary) ? ctx : NULL;
    return hipSuccess;
}

CUresult cuCtxPushCurrent_f(CUcontext ctx)
{
    CUcontext prev_ctx = NULL;
    hipError_t result = cuCtxGetCurrent_f(&prev_ctx);
    if (result != hipSuccess)
        return result;
    g_hip_context_stack.push_back(prev_ctx);
    return cuCtxSetCurrent_f(ctx);
}

CUresult cuCtxPopCurrent_f(CUcontext* ctx)
{
    if (g_hip_context_stack.empty())
        return hipErrorInvalidContext;

    if (ctx) {
        hipError_t result = cuCtxGetCurrent_f(ctx);
        if (result != hipSuccess)
            return result;
    }

    CUcontext prev_ctx = g_hip_context_stack.back();
    g_hip_context_stack.pop_back();
    return cuCtxSetCurrent_f(prev_ctx);
}

CUresult cuCtxSynchronize_f() { return hipDeviceSynchronize(); }

CUresult cuCtxGetDevice_f(CUdevice* dev) { return hipGetDevice(dev); }

CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev)
{
    (void)flags;
    if (!get_primary_context(dev))
        return hipErrorInvalidDevice;
    *ctx = new WpHipContext { dev, false };
    return cuCtxSetCurrent_f(*ctx);
}

CUresult cuCtxDestroy_f(CUcontext ctx)
{
    if (!ctx)
        return hipErrorInvalidContext;
    if (ctx == g_hip_current_context)
        g_hip_current_context = NULL;
    if (!ctx->is_primary)
        delete ctx;
    return hipSuccess;
}

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
    return hipDeviceEnablePeerAccess(peer_ctx->device, flags);
}

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx) { return hipDeviceDisablePeerAccess(peer_ctx->device); }

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags) { return hipStreamCreateWithFlags(stream, flags); }

CUresult cuStreamDestroy_f(CUstream stream) { return hipStreamDestroy(stream); }

CUresult cuStreamQuery_f(CUstream stream) { return hipStreamQuery(stream); }

CUresult cuStreamSynchronize_f(CUstream stream) { return hipStreamSynchronize(stream); }

CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags)
{
    return hipStreamWaitEvent(stream, event, flags);
}

CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx)
{
    hipDevice_t device;
    hipError_t result = hipStreamGetDevice(stream, &device);
    if (result != hipSuccess)
        return result;

    // a stream belongs to its device; report the current context when it is on that device
    CUcontext current = NULL;
    if (cuCtxGetCurrent_f(&current) == hipSuccess && current && current->device == device)
        *pctx = current;
    else
        *pctx = get_primary_context(device);

    return hipSuccess;
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
    static_assert(sizeof(cuuint64_t) == sizeof(unsigned long long), "unexpected capture id size");
    return hipStreamGetCaptureInfo_v2(
        stream, captureStatus_out, reinterpret_cast<unsigned long long*>(id_out), graph_out, dependencies_out,
        numDependencies_out
    );
}

CUresult cuStreamUpdateCaptureDependencies_f(
    CUstream stream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags
)
{
    return hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
}

CUresult cuStreamCreateWithPriority_f(CUstream* phStream, unsigned int flags, int priority)
{
    return hipStreamCreateWithPriority(phStream, flags, priority);
}

CUresult cuStreamGetPriority_f(CUstream hStream, int* priority) { return hipStreamGetPriority(hStream, priority); }

CUresult cuEventCreate_f(CUevent* event, unsigned int flags) { return hipEventCreateWithFlags(event, flags); }

CUresult cuEventDestroy_f(CUevent event) { return hipEventDestroy(event); }

CUresult cuEventQuery_f(CUevent event) { return hipEventQuery(event); }

CUresult cuEventRecord_f(CUevent event, CUstream stream) { return hipEventRecord(event, stream); }

CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
    return hipEventRecordWithFlags(event, stream, flags);
}

CUresult cuEventSynchronize_f(CUevent event) { return hipEventSynchronize(event); }

CUresult cuGraphNodeGetDependentNodes_f(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
    return hipGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
}

CUresult cuGraphNodeGetType_f(CUgraphNode hNode, CUgraphNodeType* type) { return hipGraphNodeGetType(hNode, type); }

CUresult cuModuleLoadDataEx_f(
    CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues
)
{
    return hipModuleLoadDataEx(module, image, numOptions, options, optionValues);
}

CUresult cuModuleUnload_f(CUmodule hmod) { return hipModuleUnload(hmod); }

CUresult cuModuleGetFunction_f(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    return hipModuleGetFunction(hfunc, hmod, name);
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
    return hipModuleLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra
    );
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
    // HIP has no variant taking a block-size-to-shared-memory callback
    if (blockSizeToDynamicSMemSize)
        return hipErrorNotSupported;

    // hipModuleOccupancyMaxPotentialBlockSize does not honour the kernel's own maximum block size
    // (__launch_bounds__), so pass it as the limit
    int max_threads = 0;
    hipError_t result = hipFuncGetAttribute(&max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
    if (result != hipSuccess)
        return result;
    if (blockSizeLimit <= 0 || blockSizeLimit > max_threads)
        blockSizeLimit = max_threads;

    return hipModuleOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, dynamicSMemSize, blockSizeLimit);
}

CUresult cuMemcpyPeerAsync_f(
    CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream
)
{
    return hipMemcpyPeerAsync(dst_ptr, dst_ctx->device, src_ptr, src_ctx->device, n, stream);
}

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    return hipPointerGetAttribute(data, attribute, ptr);
}

CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream)
{
    return hipGraphicsMapResources(int(count), resources, stream);
}

CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return hipGraphicsUnmapResources(int(count), resources, hStream);
}

CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    return hipGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource);
}

CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource* pCudaResource, unsigned int buffer, unsigned int flags)
{
    return hipGraphicsGLRegisterBuffer(pCudaResource, buffer, flags);
}

CUresult cuGraphicsGLRegisterImage_f(
    CUgraphicsResource* pCudaResource, unsigned int image, unsigned int target, unsigned int flags
)
{
    return hipGraphicsGLRegisterImage(pCudaResource, image, target, flags);
}

CUresult cuGraphicsSubResourceGetMappedArray_f(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel
)
{
    return hipGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
}

CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource) { return hipGraphicsUnregisterResource(resource); }

CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    (void)hfunc;

    // AMD GPUs have no opt-in for dynamic shared memory: a kernel may use up to the
    // per-block limit without configuration
    if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        hipDevice_t device;
        int max_smem_bytes = 0;
        hipError_t result = hipGetDevice(&device);
        if (result == hipSuccess)
            result = hipDeviceGetAttribute(&max_smem_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
        if (result != hipSuccess)
            return result;
        return value <= max_smem_bytes ? hipSuccess : hipErrorInvalidValue;
    }

    return hipErrorNotSupported;
}

CUresult cuIpcGetEventHandle_f(CUipcEventHandle* pHandle, CUevent event)
{
    return hipIpcGetEventHandle(pHandle, event);
}

CUresult cuIpcOpenEventHandle_f(CUevent* phEvent, CUipcEventHandle handle)
{
    return hipIpcOpenEventHandle(phEvent, handle);
}

CUresult cuIpcGetMemHandle_f(CUipcMemHandle* pHandle, CUdeviceptr dptr) { return hipIpcGetMemHandle(pHandle, dptr); }

CUresult cuIpcOpenMemHandle_f(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags)
{
    return hipIpcOpenMemHandle(pdptr, handle, flags);
}

CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr) { return hipIpcCloseMemHandle(dptr); }

// Texture functions
CUresult cuArrayCreate_f(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
    return hipArrayCreate(pHandle, pAllocateArray);
}

CUresult cuArrayDestroy_f(CUarray hArray) { return hipArrayDestroy(hArray); }

CUresult cuArray3DCreate_f(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    return hipArray3DCreate(pHandle, pAllocateArray);
}

CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    return hipArray3DGetDescriptor(pArrayDescriptor, hArray);
}

CUresult cuMemcpy2D_f(const CUDA_MEMCPY2D* pCopy) { return hipMemcpyParam2D(pCopy); }

CUresult cuMemcpy2DAsync_f(const CUDA_MEMCPY2D* pCopy, CUstream hStream)
{
    return hipMemcpyParam2DAsync(pCopy, hStream);
}

CUresult cuMemcpy3D_f(const CUDA_MEMCPY3D* pCopy) { return hipDrvMemcpy3D(pCopy); }

CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream) { return hipDrvMemcpy3DAsync(pCopy, hStream); }

CUresult cuTexObjectCreate_f(
    CUtexObject* pTexObject,
    const CUDA_RESOURCE_DESC* pResDesc,
    const CUDA_TEXTURE_DESC* pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc
)
{
    return hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

CUresult cuTexObjectDestroy_f(CUtexObject texObject) { return hipTexObjectDestroy(texObject); }

#else

// the minimum CUDA version required from the driver
#define WP_CUDA_DRIVER_VERSION 12000

// the minimum CUDA Toolkit version required to build Warp
#define WP_CUDA_TOOLKIT_VERSION 12000

// check if the CUDA Toolkit is too old
#if CUDA_VERSION < WP_CUDA_TOOLKIT_VERSION
#error Building Warp requires CUDA Toolkit version 12.0 or higher
#endif

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


// function pointers to driver API entry points
// these are explicitly versioned according to cudaTypedefs.h from CUDA Toolkit WP_CUDA_TOOLKIT_VERSION

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
static PFN_cuIpcGetEventHandle_v4010 pfn_cuIpcGetEventHandle;
static PFN_cuIpcOpenEventHandle_v4010 pfn_cuIpcOpenEventHandle;
static PFN_cuIpcGetMemHandle_v4010 pfn_cuIpcGetMemHandle;
static PFN_cuIpcOpenMemHandle_v11000 pfn_cuIpcOpenMemHandle;
static PFN_cuIpcCloseMemHandle_v4010 pfn_cuIpcCloseMemHandle;

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

static bool cuda_driver_initialized = false;

bool ContextGuard::always_restore = false;

CudaTimingState* g_cuda_timing_state = NULL;


static inline int get_major(int version) { return version / 1000; }

static inline int get_minor(int version) { return (version % 1000) / 10; }

// Get versioned driver entry point. The version argument should match the function pointer type.
// For example, to initialize PFN_cuCtxCreate_v3020 use version 3020.
static bool get_driver_entry_point(const char* name, int version, void** pfn)
{
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
}

bool init_cuda_driver()
{
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
    get_driver_entry_point("cuIpcGetEventHandle", 4010, &(void*&)pfn_cuIpcGetEventHandle);
    get_driver_entry_point("cuIpcOpenEventHandle", 4010, &(void*&)pfn_cuIpcOpenEventHandle);
    get_driver_entry_point("cuIpcGetMemHandle", 4010, &(void*&)pfn_cuIpcGetMemHandle);
    get_driver_entry_point("cuIpcOpenMemHandle", 11000, &(void*&)pfn_cuIpcOpenMemHandle);
    get_driver_entry_point("cuIpcCloseMemHandle", 4010, &(void*&)pfn_cuIpcCloseMemHandle);

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

    if (pfn_cuInit)
        cuda_driver_initialized = check_cu(pfn_cuInit(0));

    return cuda_driver_initialized;
}

#endif  // WP_ENABLE_HIP

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

    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);

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

#if !WP_ENABLE_HIP

#define DRIVER_ENTRY_POINT_ERROR driver_entry_point_error(__FUNCTION__)

static CUresult driver_entry_point_error(const char* function)
{
    fprintf(stderr, "Warp CUDA error: Function %s: a suitable driver entry point was not found\n", function);
    return (CUresult)cudaErrorCallRequiresNewerDriver;  // this matches what cudart would do
}

CUresult cuDriverGetVersion_f(int* version)
{
    return pfn_cuDriverGetVersion ? pfn_cuDriverGetVersion(version) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorName ? pfn_cuGetErrorName(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorString ? pfn_cuGetErrorString(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuInit_f(unsigned int flags) { return pfn_cuInit ? pfn_cuInit(flags) : DRIVER_ENTRY_POINT_ERROR; }

CUresult cuDeviceGet_f(CUdevice* dev, int ordinal)
{
    return pfn_cuDeviceGet ? pfn_cuDeviceGet(dev, ordinal) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetCount_f(int* count)
{
    if (pfn_cuDeviceGetCount)
        return pfn_cuDeviceGetCount(count);

    // allow calling this function even if CUDA is not available
    if (count)
        *count = 0;

    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev)
{
    return pfn_cuDeviceGetName ? pfn_cuDeviceGetName(name, len, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return pfn_cuDeviceGetAttribute ? pfn_cuDeviceGetAttribute(value, attrib, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev)
{
    return pfn_cuDeviceGetUuid ? pfn_cuDeviceGetUuid(uuid, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRetain ? pfn_cuDevicePrimaryCtxRetain(ctx, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRelease ? pfn_cuDevicePrimaryCtxRelease(dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
    return pfn_cuDeviceCanAccessPeer ? pfn_cuDeviceCanAccessPeer(can_access, dev, peer_dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemGetInfo_f(size_t* free, size_t* total)
{
    return pfn_cuMemGetInfo ? pfn_cuMemGetInfo(free, total) : DRIVER_ENTRY_POINT_ERROR;
}

#if CUDA_VERSION >= 12080
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
    return pfn_cuMemcpyBatchAsync
        ? pfn_cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, hStream)
        : DRIVER_ENTRY_POINT_ERROR;
}
#endif

CUresult cuCtxGetCurrent_f(CUcontext* ctx)
{
    return pfn_cuCtxGetCurrent ? pfn_cuCtxGetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSetCurrent_f(CUcontext ctx)
{
    return pfn_cuCtxSetCurrent ? pfn_cuCtxSetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxPushCurrent_f(CUcontext ctx)
{
    return pfn_cuCtxPushCurrent ? pfn_cuCtxPushCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxPopCurrent_f(CUcontext* ctx)
{
    return pfn_cuCtxPopCurrent ? pfn_cuCtxPopCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSynchronize_f() { return pfn_cuCtxSynchronize ? pfn_cuCtxSynchronize() : DRIVER_ENTRY_POINT_ERROR; }

CUresult cuCtxGetDevice_f(CUdevice* dev)
{
    return pfn_cuCtxGetDevice ? pfn_cuCtxGetDevice(dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev)
{
    return pfn_cuCtxCreate ? pfn_cuCtxCreate(ctx, flags, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxDestroy_f(CUcontext ctx) { return pfn_cuCtxDestroy ? pfn_cuCtxDestroy(ctx) : DRIVER_ENTRY_POINT_ERROR; }

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
    return pfn_cuCtxEnablePeerAccess ? pfn_cuCtxEnablePeerAccess(peer_ctx, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx)
{
    return pfn_cuCtxDisablePeerAccess ? pfn_cuCtxDisablePeerAccess(peer_ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags)
{
    return pfn_cuStreamCreate ? pfn_cuStreamCreate(stream, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamDestroy_f(CUstream stream)
{
    return pfn_cuStreamDestroy ? pfn_cuStreamDestroy(stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamQuery_f(CUstream stream)
{
    return pfn_cuStreamQuery ? pfn_cuStreamQuery(stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamSynchronize_f(CUstream stream)
{
    return pfn_cuStreamSynchronize ? pfn_cuStreamSynchronize(stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags)
{
    return pfn_cuStreamWaitEvent ? pfn_cuStreamWaitEvent(stream, event, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx)
{
    return pfn_cuStreamGetCtx ? pfn_cuStreamGetCtx(stream, pctx) : DRIVER_ENTRY_POINT_ERROR;
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
    return pfn_cuStreamGetCaptureInfo
        ? pfn_cuStreamGetCaptureInfo(
              stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out
          )
        : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamUpdateCaptureDependencies_f(
    CUstream stream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags
)
{
    return pfn_cuStreamUpdateCaptureDependencies
        ? pfn_cuStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)
        : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamCreateWithPriority_f(CUstream* phStream, unsigned int flags, int priority)
{
    return pfn_cuStreamCreateWithPriority ? pfn_cuStreamCreateWithPriority(phStream, flags, priority)
                                          : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuStreamGetPriority_f(CUstream hStream, int* priority)
{
    return pfn_cuStreamGetPriority ? pfn_cuStreamGetPriority(hStream, priority) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventCreate_f(CUevent* event, unsigned int flags)
{
    return pfn_cuEventCreate ? pfn_cuEventCreate(event, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventDestroy_f(CUevent event)
{
    return pfn_cuEventDestroy ? pfn_cuEventDestroy(event) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventQuery_f(CUevent event) { return pfn_cuEventQuery ? pfn_cuEventQuery(event) : DRIVER_ENTRY_POINT_ERROR; }

CUresult cuEventRecord_f(CUevent event, CUstream stream)
{
    return pfn_cuEventRecord ? pfn_cuEventRecord(event, stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
    return pfn_cuEventRecordWithFlags ? pfn_cuEventRecordWithFlags(event, stream, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuEventSynchronize_f(CUevent event)
{
    return pfn_cuEventSynchronize ? pfn_cuEventSynchronize(event) : DRIVER_ENTRY_POINT_ERROR;
}

#if CUDA_VERSION >= 12030
CUresult cuGraphAddNode_f(
    CUgraphNode* phGraphNode,
    CUgraph hGraph,
    const CUgraphNode* dependencies,
    const CUgraphEdgeData* dependencyData,
    size_t numDependencies,
    CUgraphNodeParams* nodeParams
)
{
    return pfn_cuGraphAddNode
        ? pfn_cuGraphAddNode(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)
        : DRIVER_ENTRY_POINT_ERROR;
}
#endif

CUresult cuGraphNodeGetDependentNodes_f(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
    return pfn_cuGraphNodeGetDependentNodes ? pfn_cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
                                            : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphNodeGetType_f(CUgraphNode hNode, CUgraphNodeType* type)
{
    return pfn_cuGraphNodeGetType ? pfn_cuGraphNodeGetType(hNode, type) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleLoadDataEx_f(
    CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues
)
{
    return pfn_cuModuleLoadDataEx ? pfn_cuModuleLoadDataEx(module, image, numOptions, options, optionValues)
                                  : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleUnload_f(CUmodule hmod)
{
    return pfn_cuModuleUnload ? pfn_cuModuleUnload(hmod) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleGetFunction_f(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    return pfn_cuModuleGetFunction ? pfn_cuModuleGetFunction(hfunc, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
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
    return pfn_cuLaunchKernel ? pfn_cuLaunchKernel(
                                    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                                    hStream, kernelParams, extra
                                )
                              : DRIVER_ENTRY_POINT_ERROR;
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
    return pfn_cuOccupancyMaxPotentialBlockSize
        ? pfn_cuOccupancyMaxPotentialBlockSize(
              minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit
          )
        : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpyPeerAsync_f(
    CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr, CUcontext src_ctx, size_t n, CUstream stream
)
{
    return pfn_cuMemcpyPeerAsync ? pfn_cuMemcpyPeerAsync(dst_ptr, dst_ctx, src_ptr, src_ctx, n, stream)
                                 : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    return pfn_cuPointerGetAttribute ? pfn_cuPointerGetAttribute(data, attribute, ptr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream)
{
    return pfn_cuGraphicsMapResources ? pfn_cuGraphicsMapResources(count, resources, stream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return pfn_cuGraphicsUnmapResources ? pfn_cuGraphicsUnmapResources(count, resources, hStream)
                                        : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    return pfn_cuGraphicsResourceGetMappedPointer ? pfn_cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource)
                                                  : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource* pCudaResource, unsigned int buffer, unsigned int flags)
{
    return pfn_cuGraphicsGLRegisterBuffer ? pfn_cuGraphicsGLRegisterBuffer(pCudaResource, (wp::GLuint)buffer, flags)
                                          : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsGLRegisterImage_f(
    CUgraphicsResource* pCudaResource, unsigned int image, unsigned int target, unsigned int flags
)
{
    return pfn_cuGraphicsGLRegisterImage ? pfn_cuGraphicsGLRegisterImage(pCudaResource, image, target, flags)
                                         : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsSubResourceGetMappedArray_f(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel
)
{
    return pfn_cuGraphicsSubResourceGetMappedArray
        ? pfn_cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
        : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource)
{
    return pfn_cuGraphicsUnregisterResource ? pfn_cuGraphicsUnregisterResource(resource) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    return pfn_cuModuleGetGlobal ? pfn_cuModuleGetGlobal(dptr, bytes, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    return pfn_cuFuncSetAttribute ? pfn_cuFuncSetAttribute(hfunc, attrib, value) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuIpcGetEventHandle_f(CUipcEventHandle* pHandle, CUevent event)
{
    return pfn_cuIpcGetEventHandle ? pfn_cuIpcGetEventHandle(pHandle, event) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuIpcOpenEventHandle_f(CUevent* phEvent, CUipcEventHandle handle)
{
    return pfn_cuIpcOpenEventHandle ? pfn_cuIpcOpenEventHandle(phEvent, handle) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuIpcGetMemHandle_f(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
    return pfn_cuIpcGetMemHandle ? pfn_cuIpcGetMemHandle(pHandle, dptr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuIpcOpenMemHandle_f(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags)
{
    return pfn_cuIpcOpenMemHandle ? pfn_cuIpcOpenMemHandle(pdptr, handle, flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr)
{
    return pfn_cuIpcCloseMemHandle ? pfn_cuIpcCloseMemHandle(dptr) : DRIVER_ENTRY_POINT_ERROR;
}

// Texture functions
CUresult cuArrayCreate_f(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
    return pfn_cuArrayCreate ? pfn_cuArrayCreate(pHandle, pAllocateArray) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuArrayDestroy_f(CUarray hArray)
{
    return pfn_cuArrayDestroy ? pfn_cuArrayDestroy(hArray) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuArray3DCreate_f(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    return pfn_cuArray3DCreate ? pfn_cuArray3DCreate(pHandle, pAllocateArray) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    return pfn_cuArray3DGetDescriptor ? pfn_cuArray3DGetDescriptor(pArrayDescriptor, hArray) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpy2D_f(const CUDA_MEMCPY2D* pCopy)
{
    return pfn_cuMemcpy2D ? pfn_cuMemcpy2D(pCopy) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpy2DAsync_f(const CUDA_MEMCPY2D* pCopy, CUstream hStream)
{
    return pfn_cuMemcpy2DAsync ? pfn_cuMemcpy2DAsync(pCopy, hStream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpy3D_f(const CUDA_MEMCPY3D* pCopy)
{
    return pfn_cuMemcpy3D ? pfn_cuMemcpy3D(pCopy) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream)
{
    return pfn_cuMemcpy3DAsync ? pfn_cuMemcpy3DAsync(pCopy, hStream) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuTexObjectCreate_f(
    CUtexObject* pTexObject,
    const CUDA_RESOURCE_DESC* pResDesc,
    const CUDA_TEXTURE_DESC* pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc
)
{
    return pfn_cuTexObjectCreate ? pfn_cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
                                 : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuTexObjectDestroy_f(CUtexObject texObject)
{
    return pfn_cuTexObjectDestroy ? pfn_cuTexObjectDestroy(texObject) : DRIVER_ENTRY_POINT_ERROR;
}

#endif  // !WP_ENABLE_HIP

#endif  // WP_ENABLE_CUDA
