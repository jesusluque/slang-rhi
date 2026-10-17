#pragma once

#include "metal-base.h"

namespace rhi::metal {

class BufferImpl : public Buffer
{
public:
    BufferImpl(Device* device, const BufferDesc& desc);
    ~BufferImpl();

    virtual void deleteThis() override;

    // IResource implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;

    // IBuffer implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(NativeHandle* outHandle) override;
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

public:
    NS::SharedPtr<MTL::Buffer> m_buffer;
    DeviceAddress m_deviceAddress = 0;
    /// Command buffer uploading the initial data, if it may still be executing.
    /// Work on the device's queue is ordered after it, but the native buffer
    /// can be used elsewhere, so getNativeHandle() waits for it.
    NS::SharedPtr<MTL::CommandBuffer> m_pendingUpload;
    BufferImpl* m_nextAtSameAddr = nullptr;
};

} // namespace rhi::metal
