#pragma once

#include "../base/gpu-profiler.h"
#include "../base/logo-scene.h"
#include "settings.h"
#include <slang-rhi/shader-cursor.h>

namespace rhi::particles {

std::vector<math::float4> createLogoTargets();

struct Programs
{
    ComPtr<IShaderProgram> simulate, spawn, finalize, particles, logo, bloom, toneMap;
};

struct Particle
{
    math::float4 positionAge, velocityLife, colorSize, targetStyle;
};
static_assert(sizeof(Particle) == 64);

struct SimulationParams
{
    uint32_t capacity = 0, spawnCount = 400, tick = 0;
    float dt = 1.0f / 120.0f;
    math::float2 mouse = {};
    float mouseForce = 0;
    uint32_t cycleTick = 0;
    math::float2 mousePrevious = {}, mouseVelocity = {};
};
static_assert(sizeof(SimulationParams) == 48);

class Renderer
{
public:
    static constexpr uint32_t kDefaultCapacity = 262144;

    Result init(IDevice* device, Format outputFormat, const Programs& programs, uint32_t capacity = kDefaultCapacity);

    Result resize(uint32_t width, uint32_t height);

    void reset(ICommandEncoder* encoder);

    Result step(ICommandEncoder* encoder, SimulationParams params, GpuProfiler* profiler = nullptr);

    Result render(ICommandEncoder* encoder, ITexture* output, bool bloom = true, GpuProfiler* profiler = nullptr);

    IBuffer* countBuffer() const;
    IBuffer* particleBuffer() const;
    IBuffer* argumentBuffer() const;
    bool indirect() const;
    void setIndirect(bool enabled);
    uint32_t capacity() const;
    void burst();
    const char* phaseName() const;

private:
    RenderState renderState() const;

    ComPtr<IDevice> m_device;
    ComPtr<IBuffer> m_particles[2], m_counts[2], m_arguments, m_logoVertices, m_logoIndices;
    ComPtr<IBuffer> m_logoTargets;
    ComPtr<IComputePipeline> m_simulate, m_spawn, m_finalize, m_bloomPipeline;
    ComPtr<IRenderPipeline> m_particlePipeline, m_logoPipeline, m_toneMapPipeline;
    ComPtr<ISampler> m_sampler;
    ComPtr<ITexture> m_hdr, m_bloom[2];
    Format m_outputFormat = Format::Undefined;
    uint32_t m_capacity = 0, m_current = 0, m_tick = 0, m_width = 0, m_height = 0;
    uint32_t m_cycleTick = 0;
    float m_renderCycleTime = 0;
    bool m_indirect = true;
};

} // namespace rhi::particles
