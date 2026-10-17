#include "metal-query.h"
#include "metal-device.h"
#include "metal-command.h"
#include "metal-utils.h"

namespace rhi::metal {

QueryPoolImpl::QueryPoolImpl(Device* device, const QueryPoolDesc& desc)
    : QueryPool(device, desc)
{
}

QueryPoolImpl::~QueryPoolImpl() {}

MTL::CounterSet* findCounterSet(MTL::Device* device, QueryType queryType)
{
    if (queryType != QueryType::Timestamp)
    {
        return nullptr;
    }

    for (int i = 0; i < device->counterSets()->count(); ++i)
    {
        MTL::CounterSet* counterSet = static_cast<MTL::CounterSet*>(device->counterSets()->object(i));
        if (!counterSet->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
        {
            continue;
        }
        for (int j = 0; j < counterSet->counters()->count(); ++j)
        {
            MTL::Counter* counter = static_cast<MTL::Counter*>(counterSet->counters()->object(j));
            if (counter->name()->isEqualToString(MTL::CommonCounterTimestamp))
            {
                return counterSet;
            }
        }
    }
    return nullptr;
}

Result QueryPoolImpl::init()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    MTL::CounterSet* counterSet = findCounterSet(device->m_device.get(), m_desc.type);
    if (!counterSet)
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    NS::SharedPtr<MTL::CounterSampleBufferDescriptor> counterSampleBufferDesc =
        NS::TransferPtr(MTL::CounterSampleBufferDescriptor::alloc()->init());
    counterSampleBufferDesc->setStorageMode(MTL::StorageModeShared);
    counterSampleBufferDesc->setSampleCount(m_desc.count);
    counterSampleBufferDesc->setCounterSet(counterSet);
    if (m_desc.label)
    {
        counterSampleBufferDesc->setLabel(createString(m_desc.label).get());
    }

    NS::Error* error;
    m_counterSampleBuffer =
        NS::TransferPtr(device->m_device->newCounterSampleBuffer(counterSampleBufferDesc.get(), &error));

    return m_counterSampleBuffer ? SLANG_OK : SLANG_FAIL;
}

Result QueryPoolImpl::getResultState(uint32_t queryIndex, uint32_t count, QueryResultState* outState)
{
    if (!outState || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    if (count == 0)
    {
        *outState = QueryResultState::Resolved;
        return SLANG_OK;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state != QueryResultState::Pending)
    {
        *outState = queryInfo.state;
        return SLANG_OK;
    }

    DeviceImpl* device = getDevice<DeviceImpl>();
    if (device->m_queue->updateLastFinishedID() < queryInfo.submissionID)
    {
        *outState = QueryResultState::Pending;
        return SLANG_OK;
    }

    markQueryRangeResolved(queryIndex, count, queryInfo.submissionID);
    *outState = QueryResultState::Resolved;
    return SLANG_OK;
}

Result QueryPoolImpl::getResult(uint32_t queryIndex, uint32_t count, uint64_t* outData)
{
    AUTORELEASEPOOL

    if (!outData || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    if (count == 0)
    {
        return SLANG_OK;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state == QueryResultState::Reset)
    {
        return SLANG_FAIL;
    }

    // Wait for the submission that wrote the queries to finish.
    DeviceImpl* device = getDevice<DeviceImpl>();
    if (queryInfo.state == QueryResultState::Pending &&
        device->m_queue->updateLastFinishedID() < queryInfo.submissionID)
    {
        SLANG_RETURN_ON_FAIL(device->m_queue->waitOnHost());
    }

    NS::Data* data = m_counterSampleBuffer->resolveCounterRange(NS::Range(queryIndex, count));
    if (!data || data->length() < count * sizeof(uint64_t))
    {
        return SLANG_FAIL;
    }
    ::memcpy(outData, data->mutableBytes(), count * sizeof(uint64_t));

    markQueryRangeResolved(queryIndex, count, queryInfo.submissionID);
    return SLANG_OK;
}

} // namespace rhi::metal
