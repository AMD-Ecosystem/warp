// SPDX-License-Identifier: Apache-2.0

// CUDA-to-HIP compatibility layer for ROCm builds (WP_ENABLE_HIP=1).
//
// The native sources keep their CUDA spelling. On ROCm builds the headers under
// native/hip/ shadow the CUDA headers they include (cuda.h, cuda_runtime_api.h,
// cub/..., nvrtc.h) and pull in this file, which maps the CUDA names Warp uses
// onto HIP. Driver API entry points are not mapped here: cuda_util.cpp
// implements its cu*_f wrappers directly on HIP.
//
// CUDA_VERSION and CUDART_VERSION are deliberately left undefined: hipCUB and
// rocPRIM read them as "NVIDIA backend", and every CUDA version gate in Warp is
// decided explicitly for HIP instead.

#pragma once

#if !WP_ENABLE_HIP
#error "hip_compat.h is only used by ROCm builds"
#endif

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include <hip/hiprtc.h>

#define CUDAAPI
#define CUDART_CB

// Warp contexts on HIP. HIP has no driver-API context objects (the hipCtx API is
// deprecated), so a context is a per-device handle owned by cuda_util.cpp.
struct WpHipContext;
typedef WpHipContext* CUcontext;

typedef uint64_t cuuint64_t;
typedef size_t (*CUoccupancyB2DSize)(int blockSize);

// the C++ overload of hipHostMalloc defaults its flags to hipHostMallocDefault
#define cudaMallocHost hipHostMalloc

#define CU_EVENT_WAIT_DEFAULT hipEventWaitDefault
#define CU_EVENT_WAIT_EXTERNAL hipEventWaitExternal
#define cudaErrorCallRequiresNewerDriver hipErrorNotSupported

// Runtime API
#define cudaArray_t                                 hipArray_t
#define cudaCpuDeviceId                             hipCpuDeviceId
#define cudaCreateSurfaceObject                     hipCreateSurfaceObject
#define cudaDestroySurfaceObject                    hipDestroySurfaceObject
#define cudaDeviceCanAccessPeer                     hipDeviceCanAccessPeer
#define cudaDeviceGetDefaultMemPool                 hipDeviceGetDefaultMemPool
#define cudaDeviceGetGraphMemAttribute              hipDeviceGetGraphMemAttribute
#define cudaDeviceGraphMemTrim                      hipDeviceGraphMemTrim
#define cudaDeviceProp                              hipDeviceProp_t
#define cudaDeviceSynchronize                       hipDeviceSynchronize
#define cudaErrorInvalidValue                       hipErrorInvalidValue
#define cudaError_t                                 hipError_t
#define cudaEventCreate                             hipEventCreate
#define cudaEventDestroy                            hipEventDestroy
#define cudaEventElapsedTime                        hipEventElapsedTime
#define cudaEventRecord                             hipEventRecord
#define cudaEventSynchronize                        hipEventSynchronize
#define cudaEvent_t                                 hipEvent_t
#define cudaFree                                    hipFree
#define cudaFreeAsync                               hipFreeAsync
#define cudaFreeHost                                hipHostFree
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute                        hipFuncSetAttribute
#define cudaGetDevice                               hipGetDevice
#define cudaGetDeviceCount                          hipGetDeviceCount
#define cudaGetDeviceProperties                     hipGetDeviceProperties
#define cudaGetErrorString                          hipGetErrorString
#define cudaGetLastError                            hipGetLastError
#define cudaGraphAddChildGraphNode                  hipGraphAddChildGraphNode
#define cudaGraphAddEmptyNode                       hipGraphAddEmptyNode
#define cudaGraphAddMemFreeNode                     hipGraphAddMemFreeNode
#define cudaGraphAddMemcpyNode1D                    hipGraphAddMemcpyNode1D
#define cudaGraphChildGraphNodeGetGraph             hipGraphChildGraphNodeGetGraph
#define cudaGraphDebugDotPrint                      hipGraphDebugDotPrint
#define cudaGraphDestroy                            hipGraphDestroy
#define cudaGraphExecDestroy                        hipGraphExecDestroy
#define cudaGraphExecMemcpyNodeSetParams1D          hipGraphExecMemcpyNodeSetParams1D
#define cudaGraphExec_t                             hipGraphExec_t
#define cudaGraphGetNodes                           hipGraphGetNodes
#define cudaGraphInstantiateFlagAutoFreeOnLaunch    hipGraphInstantiateFlagAutoFreeOnLaunch
#define cudaGraphInstantiateWithFlags               hipGraphInstantiateWithFlags
#if defined(_WIN32)
// On Windows the HIP runtime executes graph memory nodes on a worker thread. A launch of a graph
// instantiated with hipGraphInstantiateFlagAutoFreeOnLaunch frees the graph's memory before it
// waits for the previous launch, which may still be running those nodes, and while a memory
// allocation node runs, the runtime does not recognize pointers into that allocation, so a copy
// issued right after the launch fails with an invalid-argument error. Waiting for the launch to
// finish closes both windows (an invalid-argument error or an access violation in the runtime).
static inline hipError_t wp_hip_graph_launch(hipGraphExec_t graph_exec, hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    bool capturing = hipStreamIsCapturing(stream, &status) != hipSuccess || status != hipStreamCaptureStatusNone;
    hipError_t err = hipGraphLaunch(graph_exec, stream);
    if (err != hipSuccess || capturing)
        return err;
    return hipStreamSynchronize(stream);
}
#define cudaGraphLaunch                             wp_hip_graph_launch
#else
#define cudaGraphLaunch                             hipGraphLaunch
#endif
#define cudaGraphMemAllocNodeGetParams              hipGraphMemAllocNodeGetParams
#define cudaGraphMemAttrUsedMemCurrent              hipGraphMemAttrUsedMemCurrent
#define cudaGraphMemFreeNodeGetParams               hipGraphMemFreeNodeGetParams
#define cudaGraphNode_t                             hipGraphNode_t
#define cudaGraphRetainUserObject                   hipGraphRetainUserObject
#define cudaGraphUpload                             hipGraphUpload
#define cudaGraphUserObjectMove                     hipGraphUserObjectMove
#define cudaGraph_t                                 hipGraph_t
#define cudaInvalidDeviceId                         hipInvalidDeviceId
#define cudaMalloc                                  hipMalloc
#define cudaMallocAsync                             hipMallocAsync
#define cudaMallocManaged                           hipMallocManaged
#define cudaMemAccessDesc                           hipMemAccessDesc
#define cudaMemAccessFlags                          hipMemAccessFlags
#define cudaMemAccessFlagsProtNone                  hipMemAccessFlagsProtNone
#define cudaMemAccessFlagsProtReadWrite             hipMemAccessFlagsProtReadWrite
#define cudaMemAdvise                               hipMemAdvise
#define cudaMemAllocNodeParams                      hipMemAllocNodeParams
#define cudaMemLocation                             hipMemLocation
#define cudaMemLocationTypeDevice                   hipMemLocationTypeDevice
#define cudaMemLocationTypeHost                     hipMemLocationTypeHost
#define cudaMemLocationTypeInvalid                  hipMemLocationTypeInvalid
#define cudaMemPoolAttrReleaseThreshold             hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrUsedMemCurrent               hipMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemHigh                  hipMemPoolAttrUsedMemHigh
#define cudaMemPoolGetAccess                        hipMemPoolGetAccess
#define cudaMemPoolGetAttribute                     hipMemPoolGetAttribute
#define cudaMemPoolSetAccess                        hipMemPoolSetAccess
#define cudaMemPoolSetAttribute                     hipMemPoolSetAttribute
#define cudaMemPool_t                               hipMemPool_t
#define cudaMemPrefetchAsync                        hipMemPrefetchAsync
#define cudaMemcpy                                  hipMemcpy
#define cudaMemcpyAsync                             hipMemcpyAsync
#define cudaMemcpyDefault                           hipMemcpyDefault
#define cudaMemcpyDeviceToDevice                    hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost                      hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice                      hipMemcpyHostToDevice
#define cudaMemcpyHostToHost                        hipMemcpyHostToHost
#define cudaMemcpyKind                              hipMemcpyKind
#define cudaMemoryAdvise                            hipMemoryAdvise
#define cudaMemset                                  hipMemset
#define cudaMemsetAsync                             hipMemsetAsync
#define cudaPointerAttributes                       hipPointerAttribute_t
#define cudaPointerGetAttributes                    hipPointerGetAttributes
#define cudaResourceDesc                            hipResourceDesc
#define cudaResourceTypeArray                       hipResourceTypeArray
#define cudaSetDevice                               hipSetDevice
#define cudaStreamBeginCapture                      hipStreamBeginCapture
#define cudaStreamBeginCaptureToGraph               hipStreamBeginCaptureToGraph
#define cudaStreamCaptureMode                       hipStreamCaptureMode
#define cudaStreamCaptureModeGlobal                 hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeRelaxed                hipStreamCaptureModeRelaxed
#define cudaStreamCaptureModeThreadLocal            hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus                     hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive               hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusNone                 hipStreamCaptureStatusNone
#define cudaStreamEndCapture                        hipStreamEndCapture
#define cudaStreamIsCapturing                       hipStreamIsCapturing
#define cudaStreamSetCaptureDependencies            hipStreamSetCaptureDependencies
#define cudaStreamSynchronize                       hipStreamSynchronize
#define cudaStream_t                                hipStream_t
#define cudaSuccess                                 hipSuccess
#define cudaSurfaceObject_t                         hipSurfaceObject_t
#define cudaUserObjectCreate                        hipUserObjectCreate
#define cudaUserObjectNoDestructorSync              hipUserObjectNoDestructorSync
#define cudaUserObject_t                            hipUserObject_t

// Driver API types
#define CUaddress_mode        HIPaddress_mode
#define CUarray               hipArray_t
#define CUarray_format        hipArray_Format
#define CUdevice              hipDevice_t
#define CUdevice_attribute    hipDeviceAttribute_t
#define CUdeviceptr           hipDeviceptr_t
#define CUevent               hipEvent_t
#define CUfunction            hipFunction_t
#define CUfunction_attribute  hipFunction_attribute
#define CUgraph               hipGraph_t
#define CUgraphEdgeData       hipGraphEdgeData
#define CUgraphExec           hipGraphExec_t
#define CUgraphNode           hipGraphNode_t
#define CUgraphNodeParams     hipGraphNodeParams
#define CUgraphNodeType       hipGraphNodeType
#define CUgraphicsResource    hipGraphicsResource_t
#define CUipcEventHandle      hipIpcEventHandle_t
#define CUipcMemHandle        hipIpcMemHandle_t
#define CUjit_option          hipJitOption
#define CUmemcpyAttributes    hipMemcpyAttributes
#define CUmemorytype          hipMemoryType
#define CUmodule              hipModule_t
#define CUpointer_attribute   hipPointer_attribute
#define CUresult              hipError_t
#define CUstream              hipStream_t
#define CUstreamCaptureStatus hipStreamCaptureStatus
#define CUtexObject           hipTextureObject_t
#define CUuuid                hipUUID

// Driver API constants
#define CUDA_ARRAY3D_DESCRIPTOR                                 HIP_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_SURFACE_LDST                               hipArraySurfaceLoadStore
#define CUDA_ARRAY_DESCRIPTOR                                   HIP_ARRAY_DESCRIPTOR
#define CUDA_ERROR_NOT_READY                                    hipErrorNotReady
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED                  hipErrorPeerAccessAlreadyEnabled
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED                      hipErrorPeerAccessNotEnabled
#define CUDA_MEMCPY2D                                           hip_Memcpy2D
#define CUDA_MEMCPY3D                                           HIP_MEMCPY3D
#define CUDA_RESOURCE_DESC                                      HIP_RESOURCE_DESC
#define CUDA_RESOURCE_VIEW_DESC                                 HIP_RESOURCE_VIEW_DESC
#define CUDA_SUCCESS                                            hipSuccess
#define CUDA_TEXTURE_DESC                                       HIP_TEXTURE_DESC
#define CU_AD_FORMAT_FLOAT                                      HIP_AD_FORMAT_FLOAT
#define CU_AD_FORMAT_HALF                                       HIP_AD_FORMAT_HALF
#define CU_AD_FORMAT_SIGNED_INT16                               HIP_AD_FORMAT_SIGNED_INT16
#define CU_AD_FORMAT_SIGNED_INT32                               HIP_AD_FORMAT_SIGNED_INT32
#define CU_AD_FORMAT_SIGNED_INT8                                HIP_AD_FORMAT_SIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT16                             HIP_AD_FORMAT_UNSIGNED_INT16
#define CU_AD_FORMAT_UNSIGNED_INT32                             HIP_AD_FORMAT_UNSIGNED_INT32
#define CU_AD_FORMAT_UNSIGNED_INT8                              HIP_AD_FORMAT_UNSIGNED_INT8
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR            hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR            hipDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST hipDeviceAttributeDirectManagedMemAccessFromHost
#define CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED        hipDeviceAttributeHostNativeAtomicSupported
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                          hipDeviceAttributeIntegrated
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN   hipDeviceAttributeSharedMemPerBlockOptin
#define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED              hipDeviceAttributeMemoryPoolsSupported
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                hipDeviceAttributeMultiprocessorCount
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS              hipDeviceAttributePageableMemoryAccess
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                          hipDeviceAttributePciBusId
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                       hipDeviceAttributePciDeviceId
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                       hipDeviceAttributePciDomainId
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                  hipDeviceAttributeUnifiedAddressing
#define CU_EVENT_DEFAULT                                        hipEventDefault
#define CU_EVENT_DISABLE_TIMING                                 hipEventDisableTiming
#define CU_EVENT_RECORD_DEFAULT                                 hipEventRecordDefault
#define CU_EVENT_RECORD_EXTERNAL                                hipEventRecordExternal
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES         HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_GRAPH_NODE_TYPE_BATCH_MEM_OP                         hipGraphNodeTypeBatchMemOp
#define CU_GRAPH_NODE_TYPE_EMPTY                                hipGraphNodeTypeEmpty
#define CU_GRAPH_NODE_TYPE_EVENT_RECORD                         hipGraphNodeTypeEventRecord
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL                     hipGraphNodeTypeExtSemaphoreSignal
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT                       hipGraphNodeTypeExtSemaphoreWait
#define CU_GRAPH_NODE_TYPE_GRAPH                                hipGraphNodeTypeGraph
#define CU_GRAPH_NODE_TYPE_HOST                                 hipGraphNodeTypeHost
#define CU_GRAPH_NODE_TYPE_KERNEL                               hipGraphNodeTypeKernel
#define CU_GRAPH_NODE_TYPE_MEMCPY                               hipGraphNodeTypeMemcpy
#define CU_GRAPH_NODE_TYPE_MEMSET                               hipGraphNodeTypeMemset
#define CU_GRAPH_NODE_TYPE_MEM_ALLOC                            hipGraphNodeTypeMemAlloc
#define CU_GRAPH_NODE_TYPE_MEM_FREE                             hipGraphNodeTypeMemFree
#define CU_GRAPH_NODE_TYPE_WAIT_EVENT                           hipGraphNodeTypeWaitEvent
#define CU_IPC_HANDLE_SIZE                                      HIP_IPC_HANDLE_SIZE
#define CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS                      hipIpcMemLazyEnablePeerAccess
#define CU_JIT_ERROR_LOG_BUFFER                                 hipJitOptionErrorLogBuffer
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES                      hipJitOptionErrorLogBufferSizeBytes
#define CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE              hipMemcpyFlagPreferOverlapWithCompute
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM                       hipMemcpySrcAccessOrderStream
#define CU_MEMORYTYPE_ARRAY                                     hipMemoryTypeArray
#define CU_MEMORYTYPE_DEVICE                                    hipMemoryTypeDevice
#define CU_MEMORYTYPE_HOST                                      hipMemoryTypeHost
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE                     HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define CU_RESOURCE_TYPE_ARRAY                                  HIP_RESOURCE_TYPE_ARRAY
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES                      hipStreamAddCaptureDependencies
#define CU_STREAM_CAPTURE_STATUS_ACTIVE                         hipStreamCaptureStatusActive
#define CU_STREAM_CAPTURE_STATUS_NONE                           hipStreamCaptureStatusNone
#define CU_STREAM_DEFAULT                                       hipStreamDefault
#define CU_STREAM_SET_CAPTURE_DEPENDENCIES                      hipStreamSetCaptureDependencies
#define CU_TRSF_NORMALIZED_COORDINATES                          HIP_TRSF_NORMALIZED_COORDINATES
#define CU_TRSF_READ_AS_INTEGER                                 HIP_TRSF_READ_AS_INTEGER
#define CU_TR_ADDRESS_MODE_BORDER                               HIP_TR_ADDRESS_MODE_BORDER
#define CU_TR_ADDRESS_MODE_CLAMP                                HIP_TR_ADDRESS_MODE_CLAMP
#define CU_TR_ADDRESS_MODE_MIRROR                               HIP_TR_ADDRESS_MODE_MIRROR
#define CU_TR_ADDRESS_MODE_WRAP                                 HIP_TR_ADDRESS_MODE_WRAP
#define CU_TR_FILTER_MODE_LINEAR                                HIP_TR_FILTER_MODE_LINEAR
#define CU_TR_FILTER_MODE_POINT                                 HIP_TR_FILTER_MODE_POINT

// Runtime compiler
#define NVRTC_SUCCESS          HIPRTC_SUCCESS
#define nvrtcCompileProgram    hiprtcCompileProgram
#define nvrtcCreateProgram     hiprtcCreateProgram
#define nvrtcDestroyProgram    hiprtcDestroyProgram
#define nvrtcGetErrorString    hiprtcGetErrorString
#define nvrtcGetProgramLog     hiprtcGetProgramLog
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#define nvrtcProgram           hiprtcProgram
#define nvrtcResult            hiprtcResult
#define nvrtcVersion           hiprtcVersion
