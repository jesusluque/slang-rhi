#include "metal-heap.h"
#include "metal-device.h"
#include "metal-command.h"
#include "metal-buffer.h"
#include "metal-utils.h"

namespace rhi::metal {

HeapImpl::PageImpl::PageImpl(Heap* heap, const PageDesc& desc, BufferImpl* buffer)
    : Heap::Page(heap, desc)
    , m_buffer(buffer)
{
}

DeviceAddress HeapImpl::PageImpl::offsetToAddress(Size offset)
{
    return m_buffer->getDeviceAddress() + offset;
}

HeapImpl::HeapImpl(Device* device, const HeapDesc& desc)
    : Heap(device, desc)
{
}

HeapImpl::~HeapImpl() {}

Result HeapImpl::free(HeapAlloc allocation)
{
    DeviceImpl* deviceImpl = static_cast<DeviceImpl*>(getDevice());
    CommandQueueImpl* queue = deviceImpl->m_queue;

    // If the queue is idle the allocation can be retired immediately, otherwise
    // it is retired by flush() once the work submitted so far has finished.
    if (queue->m_lastFinishedID >= queue->m_lastSubmittedID)
    {
        return retire(allocation);
    }

    PendingFree pendingFree;
    pendingFree.allocation = allocation;
    pendingFree.submitIndex = queue->m_lastSubmittedID;
    m_pendingFrees.push_back(pendingFree);
    return SLANG_OK;
}

Result HeapImpl::flush()
{
    DeviceImpl* deviceImpl = static_cast<DeviceImpl*>(getDevice());
    uint64_t lastFinishedID = deviceImpl->m_queue->updateLastFinishedID();

    for (auto it = m_pendingFrees.begin(); it != m_pendingFrees.end();)
    {
        if (it->submitIndex > lastFinishedID)
        {
            // Pending frees are ordered by submission.
            break;
        }
        SLANG_RETURN_ON_FAIL(retire(it->allocation));
        it = m_pendingFrees.erase(it);
    }
    return SLANG_OK;
}

Result HeapImpl::allocatePage(const PageDesc& desc, Page** outPage)
{
    DeviceImpl* deviceImpl = static_cast<DeviceImpl*>(getDevice());

    // Pages are regular buffers, so they are made resident like any other buffer, including
    // when shaders reach them through a device address.
    BufferDesc bufferDesc = {};
    bufferDesc.size = desc.size;
    bufferDesc.memoryType = m_desc.memoryType;
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource |
                       BufferUsage::CopyDestination;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.label = m_desc.label;
    ComPtr<IBuffer> buffer;
    Result result = deviceImpl->createBuffer(bufferDesc, nullptr, buffer.writeRef());
    if (SLANG_FAILED(result))
    {
        return result == SLANG_FAIL ? SLANG_E_OUT_OF_MEMORY : result;
    }
    BufferImpl* bufferImpl = checked_cast<BufferImpl*>(buffer.get());

    // Metal buffers are aligned to at least the page size, which covers any alignment a
    // user should request. Treat a misaligned page as an error.
    if (bufferImpl->getDeviceAddress() % desc.alignment != 0)
    {
        return SLANG_E_INVALID_ARG;
    }

    *outPage = new PageImpl(this, desc, bufferImpl);
    return SLANG_OK;
}

Result HeapImpl::freePage(Page* page)
{
    delete static_cast<PageImpl*>(page);
    return SLANG_OK;
}

Result HeapImpl::fixUpAllocDesc(HeapAllocDesc& desc)
{
    if (!math::isPowerOf2(desc.alignment))
    {
        return SLANG_E_INVALID_ARG;
    }
    return SLANG_OK;
}

Result DeviceImpl::createHeap(const HeapDesc& desc, IHeap** outHeap)
{
    if (is_set(desc.usage, HeapUsage::Shared))
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    RefPtr<HeapImpl> heap = new HeapImpl(this, desc);
    returnComPtr(outHeap, heap);
    return SLANG_OK;
}

} // namespace rhi::metal
