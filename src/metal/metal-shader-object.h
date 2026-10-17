#pragma once

#include "metal-base.h"
#include "metal-shader-object-layout.h"

#include <vector>

namespace rhi::metal {

struct UploadAllocation;

struct BindingDataBuilder
{
    DeviceImpl* m_device;
    ArenaAllocator* m_allocator;
    BindingCache* m_bindingCache;
    BindingDataImpl* m_bindingData;

    /// Bind this object as a root shader object
    Result bindAsRoot(
        RootShaderObject* shaderObject,
        RootShaderObjectLayoutImpl* specializedLayout,
        BindingDataImpl*& outBindingData
    );

    /// Bind this object as if it was declared as a `ConstantBuffer<T>` in Slang
    Result bindAsConstantBuffer(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as if it was declared as a `ParameterBlock<T>` in Slang
    Result bindAsParameterBlock(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a value that appears in the body of another object.
    ///
    /// This case is directly used when binding an object for an interface-type
    /// sub-object range when static specialization is used. It is also used
    /// indirectly when binding sub-objects to constant buffer or parameter
    /// block ranges.
    ///
    Result bindAsValue(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind the buffer for ordinary/uniform data, if needed
    ///
    /// The `ioOffset` parameter will be updated to reflect the constant buffer
    /// register consumed by the ordinary data buffer, if one was bound.
    ///
    Result bindOrdinaryDataBufferIfNeeded(
        ShaderObject* shaderObject,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    Result writeArgumentBuffer(
        ShaderObject* shaderObject,
        ShaderObjectLayoutImpl* specializedLayout,
        UploadAllocation& outArgumentBuffer
    );

    Result writeOrdinaryDataIntoArgumentBuffer(
        slang::TypeLayoutReflection* argumentBufferTypeLayout,
        slang::TypeLayoutReflection* defaultTypeLayout,
        uint8_t* argumentBuffer,
        uint8_t* srcData
    );

    Result resolvePointerFieldResidency(ShaderObject* shaderObject, ShaderObjectLayoutImpl* specializedLayout);
};

struct BindingDataImpl : BindingData
{
    MTL::Buffer** buffers;
    NS::UInteger* bufferOffsets;
    uint32_t bufferCount;
    uint32_t bufferCapacity;

    MTL::Texture** textures;
    uint32_t textureCount;
    uint32_t textureCapacity;

    MTL::SamplerState** samplers;
    uint32_t samplerCount;

    MTL::Resource** usedResources;
    uint32_t usedResourceCount;
    uint32_t usedResourceCapacity;
    MTL::Resource** usedRWResources;
    uint32_t usedRWResourceCount;
    uint32_t usedRWResourceCapacity;

    // Root-level acceleration structures bound via setAccelerationStructure:atBufferIndex:
    MTL::AccelerationStructure** rootAccelerationStructures;
    NS::UInteger* rootAccelerationStructureSlots;
    uint32_t rootAccelerationStructureCount;
    uint32_t rootAccelerationStructureCapacity;
};

/// A sub-allocation of upload memory used for constant and argument buffers.
struct UploadAllocation
{
    BufferImpl* buffer = nullptr;
    NS::UInteger offset = 0;
    uint8_t* data = nullptr;
    uint64_t deviceAddress = 0;
};

struct BindingCache
{
    /// Buffers referenced by the binding data, kept alive until the command buffer is reset.
    std::vector<InternalRefPtr<BufferImpl>> buffers;

    /// Pages of upload memory that constant and argument buffers are sub-allocated from.
    /// Creating a Metal buffer for every bind is expensive, so the pages are kept and
    /// reused once the command buffer is reset (after the GPU has finished with it).
    std::vector<InternalRefPtr<BufferImpl>> uploadPages;
    size_t uploadPageIndex = 0;
    size_t uploadPageOffset = 0;

    Result allocateUpload(DeviceImpl* device, size_t size, UploadAllocation& outAllocation);

    void reset()
    {
        buffers.clear();
        uploadPageIndex = 0;
        uploadPageOffset = 0;
    }
};

} // namespace rhi::metal
