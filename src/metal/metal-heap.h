#pragma once

#include "metal-base.h"
#include "metal-buffer.h"
#include "../heap.h"

#include <list>

namespace rhi::metal {

class HeapImpl : public Heap
{
public:
    struct PendingFree
    {
        HeapAlloc allocation;
        uint64_t submitIndex;
    };

    class PageImpl : public Heap::Page
    {
    public:
        PageImpl(Heap* heap, const PageDesc& desc, BufferImpl* buffer);

        DeviceAddress offsetToAddress(Size offset) override;

        InternalRefPtr<BufferImpl> m_buffer;
    };

    HeapImpl(Device* device, const HeapDesc& desc);
    ~HeapImpl();

    virtual SLANG_NO_THROW Result SLANG_MCALL free(HeapAlloc allocation) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL flush() override;

    virtual Result allocatePage(const PageDesc& desc, Page** outPage) override;
    virtual Result freePage(Page* page) override;

    virtual Result fixUpAllocDesc(HeapAllocDesc& desc) override;

    std::list<PendingFree> m_pendingFrees;
};

} // namespace rhi::metal
