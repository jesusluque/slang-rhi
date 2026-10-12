#include "renderer.h"

#include <algorithm>
#include <cmath>

namespace rhi::particles {

// Area-weighted samples of the visible logo surface, in particle world space.
// Upload once; spawning assigns a persistent target independently of compaction.
std::vector<math::float4> createLogoTargets()
{
    struct Triangle
    {
        math::float3 a, b, c;
        float cumulativeArea;
        uint32_t material;
    };
    std::vector<Triangle> triangles;
    float area = 0;
    for (const auto& mesh : logo::kMeshRanges)
        for (uint32_t i = mesh.firstIndex; i < mesh.firstIndex + mesh.indexCount; i += 3)
        {
            auto a = logo::kVertices[logo::kIndices[i]].position;
            auto b = logo::kVertices[logo::kIndices[i + 1]].position;
            auto c = logo::kVertices[logo::kIndices[i + 2]].position;
            float projectedArea = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if (projectedArea <= 1e-8f || a.z + b.z + c.z <= 0)
                continue;
            area += projectedArea;
            triangles.push_back({a, b, c, area, mesh.materialIndex});
        }
    if (triangles.empty())
        return {};
    uint32_t seed = 0x51a9u;
    auto random = [&]()
    {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / 16777216.0f;
    };
    std::vector<math::float4> targets;
    targets.reserve(PARTICLE_LOGO_TARGET_COUNT);
    for (uint32_t i = 0; i < PARTICLE_LOGO_TARGET_COUNT; ++i)
    {
        // Stratify triangle selection to preserve even the smaller logo regions.
        float sampleArea = (float(i) + random()) / PARTICLE_LOGO_TARGET_COUNT * area;
        const auto& triangle = *std::lower_bound(
            triangles.begin(),
            triangles.end(),
            sampleArea,
            [](const Triangle& t, float value)
            {
                return t.cumulativeArea < value;
            }
        );
        float u = std::sqrt(random()), v = random();
        auto p = triangle.a * (1 - u) + triangle.b * (u * (1 - v)) + triangle.c * (u * v);
        targets.push_back({p.x * 0.8f, (p.y - 1.58f) * 0.8f, float(triangle.material), 0});
    }
    return targets;
}


Result Renderer::init(IDevice* device, Format outputFormat, const Programs& programs, uint32_t capacity)
{
    if (!capacity || capacity > kDefaultCapacity)
        return SLANG_E_INVALID_ARG;
    m_device = device;
    m_outputFormat = outputFormat;
    m_capacity = capacity;
    // There is no indirect-draw feature bit. This is an implementation gap
    // in the Metal backend, rather than a hardware capability restriction.
    m_indirect = device->getDeviceType() != DeviceType::Metal;
    FormatSupport support;
    SLANG_RETURN_ON_FAIL(device->getFormatSupport(Format::RGBA16Float, &support));
    if (!is_set(support, FormatSupport::RenderTarget | FormatSupport::ShaderSample | FormatSupport::ShaderUavStore))
        return SLANG_E_NOT_AVAILABLE;

    for (uint32_t i = 0; i < 2; ++i)
    {
        BufferDesc desc = {};
        desc.size = Size(capacity) * sizeof(Particle);
        desc.elementSize = sizeof(Particle);
        desc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource;
        desc.label = "Particle state";
        SLANG_RETURN_ON_FAIL(device->createBuffer(desc, nullptr, m_particles[i].writeRef()));
        desc.size = sizeof(uint32_t);
        desc.elementSize = sizeof(uint32_t);
        desc.usage |= BufferUsage::CopyDestination;
        desc.label = "Particle live count";
        uint32_t zero = 0;
        SLANG_RETURN_ON_FAIL(device->createBuffer(desc, &zero, m_counts[i].writeRef()));
    }
    BufferDesc args = {};
    args.size = sizeof(IndirectDrawArguments);
    args.elementSize = 0;
    args.usage = BufferUsage::UnorderedAccess | BufferUsage::IndirectArgument | BufferUsage::CopySource |
                 BufferUsage::CopyDestination;
    args.label = "GPU particle draw arguments";
    uint32_t initialArgs[4] = {6, 0, 0, 0};
    SLANG_RETURN_ON_FAIL(device->createBuffer(args, initialArgs, m_arguments.writeRef()));

    auto targets = createLogoTargets();
    if (targets.size() != PARTICLE_LOGO_TARGET_COUNT)
        return SLANG_FAIL;
    BufferDesc targetBuffer = {};
    targetBuffer.size = targets.size() * sizeof(targets[0]);
    targetBuffer.elementSize = sizeof(targets[0]);
    targetBuffer.usage = BufferUsage::ShaderResource;
    targetBuffer.label = "Particle logo targets";
    SLANG_RETURN_ON_FAIL(device->createBuffer(targetBuffer, targets.data(), m_logoTargets.writeRef()));

    ComputePipelineDesc compute = {};
    compute.program = programs.simulate;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(compute, m_simulate.writeRef()));
    compute.program = programs.spawn;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(compute, m_spawn.writeRef()));
    compute.program = programs.finalize;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(compute, m_finalize.writeRef()));
    compute.program = programs.bloom;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(compute, m_bloomPipeline.writeRef()));

    ColorTargetDesc target = {};
    target.format = Format::RGBA16Float;
    target.enableBlend = true;
    target.color.srcFactor = target.color.dstFactor = BlendFactor::One;
    target.alpha.srcFactor = BlendFactor::Zero;
    target.alpha.dstFactor = BlendFactor::One;
    RenderPipelineDesc pipeline = {};
    pipeline.program = programs.particles;
    pipeline.targets = &target;
    pipeline.targetCount = 1;
    pipeline.depthStencil.depthTestEnable = pipeline.depthStencil.depthWriteEnable = false;
    pipeline.rasterizer.cullMode = CullMode::None;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_particlePipeline.writeRef()));

    BufferDesc mesh = {};
    mesh.size = sizeof(logo::kVertices);
    mesh.usage = BufferUsage::VertexBuffer;
    SLANG_RETURN_ON_FAIL(device->createBuffer(mesh, logo::kVertices, m_logoVertices.writeRef()));
    mesh.size = sizeof(logo::kIndices);
    mesh.usage = BufferUsage::IndexBuffer;
    SLANG_RETURN_ON_FAIL(device->createBuffer(mesh, logo::kIndices, m_logoIndices.writeRef()));
    VertexStreamDesc stream = {sizeof(logo::Vertex), InputSlotClass::PerVertex, 0};
    InputElementDesc element = {"POSITION", 0, Format::RGB32Float, 0, 0};
    InputLayoutDesc layout = {};
    layout.vertexStreams = &stream;
    layout.vertexStreamCount = 1;
    layout.inputElements = &element;
    layout.inputElementCount = 1;
    ComPtr<IInputLayout> inputLayout;
    SLANG_RETURN_ON_FAIL(device->createInputLayout(layout, inputLayout.writeRef()));
    target.enableBlend = false;
    pipeline.program = programs.logo;
    pipeline.inputLayout = inputLayout;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_logoPipeline.writeRef()));
    target.format = outputFormat;
    pipeline.program = programs.toneMap;
    pipeline.inputLayout = nullptr;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_toneMapPipeline.writeRef()));
    SamplerDesc sampler = {};
    sampler.minFilter = sampler.magFilter = TextureFilteringMode::Linear;
    sampler.addressU = sampler.addressV = sampler.addressW = TextureAddressingMode::ClampToEdge;
    return device->createSampler(sampler, m_sampler.writeRef());
}

Result Renderer::resize(uint32_t width, uint32_t height)
{
    if (!width || !height)
        return SLANG_E_INVALID_ARG;
    TextureDesc desc = {};
    desc.size = {width, height, 1};
    desc.format = Format::RGBA16Float;
    desc.usage = TextureUsage::RenderTarget | TextureUsage::ShaderResource | TextureUsage::UnorderedAccess;
    desc.label = "Particle HDR";
    SLANG_RETURN_ON_FAIL(m_device->createTexture(desc, nullptr, m_hdr.writeRef()));
    desc.size = {(width + 1) / 2, (height + 1) / 2, 1};
    desc.label = "Particle bloom";
    for (auto& texture : m_bloom)
        SLANG_RETURN_ON_FAIL(m_device->createTexture(desc, nullptr, texture.writeRef()));
    m_width = width;
    m_height = height;
    return SLANG_OK;
}

void Renderer::reset(ICommandEncoder* encoder)
{
    for (auto& count : m_counts)
        encoder->clearBuffer(count);
    encoder->clearBuffer(m_arguments);
    m_current = 0;
    m_tick = 0;
    m_cycleTick = 0;
    m_renderCycleTime = 0;
}

Result Renderer::step(ICommandEncoder* encoder, SimulationParams params, GpuProfiler* profiler)
{
    params.capacity = m_capacity;
    params.spawnCount = std::min(params.spawnCount, m_capacity);
    params.tick = m_tick++;
    params.cycleTick = m_cycleTick;
    m_renderCycleTime = float(m_cycleTick) / 120.0f;
    m_cycleTick = (m_cycleTick + 1) % PARTICLE_CYCLE_TICKS;
    uint32_t next = 1 - m_current;
    encoder->clearBuffer(m_counts[next]);
    auto pass = encoder->beginComputePass();
    uint32_t region = profiler ? profiler->beginRegion(pass, "Simulation step") : GpuProfiler::kInvalidRegion;
    pass->pushDebugGroup("Simulate and compact particles", {});
    ShaderCursor cursor(pass->bindPipeline(m_simulate));
    SLANG_RETURN_ON_FAIL(cursor["sim"].setData(params));
    SLANG_RETURN_ON_FAIL(cursor["inputParticles"].setBinding(m_particles[m_current]));
    SLANG_RETURN_ON_FAIL(cursor["inputCount"].setBinding(m_counts[m_current]));
    SLANG_RETURN_ON_FAIL(cursor["outputParticles"].setBinding(m_particles[next]));
    SLANG_RETURN_ON_FAIL(cursor["outputCount"].setBinding(m_counts[next]));
    pass->dispatchCompute((m_capacity + 127) / 128, 1, 1);
    pass->popDebugGroup();
    pass->end();

    pass = encoder->beginComputePass();
    pass->pushDebugGroup("Spawn particles", {});
    cursor = ShaderCursor(pass->bindPipeline(m_spawn));
    SLANG_RETURN_ON_FAIL(cursor["sim"].setData(params));
    SLANG_RETURN_ON_FAIL(cursor["logoTargets"].setBinding(m_logoTargets));
    SLANG_RETURN_ON_FAIL(cursor["outputParticles"].setBinding(m_particles[next]));
    SLANG_RETURN_ON_FAIL(cursor["outputCount"].setBinding(m_counts[next]));
    pass->dispatchCompute(std::max(1u, (params.spawnCount + 127) / 128), 1, 1);
    pass->popDebugGroup();
    pass->end();

    pass = encoder->beginComputePass();
    pass->pushDebugGroup("Generate particle draw arguments", {});
    cursor = ShaderCursor(pass->bindPipeline(m_finalize));
    SLANG_RETURN_ON_FAIL(cursor["sim"].setData(params));
    SLANG_RETURN_ON_FAIL(cursor["outputCount"].setBinding(m_counts[next]));
    SLANG_RETURN_ON_FAIL(cursor["drawArguments"].setBinding(m_arguments));
    pass->dispatchCompute(1, 1, 1);
    if (profiler)
        profiler->endRegion(pass, region);
    pass->popDebugGroup();
    pass->end();
    m_current = next;
    return SLANG_OK;
}

Result Renderer::render(ICommandEncoder* encoder, ITexture* output, bool bloom, GpuProfiler* profiler)
{
    if (!m_hdr || output->getDesc().size.width != m_width || output->getDesc().size.height != m_height ||
        output->getDesc().format != m_outputFormat)
        return SLANG_E_INVALID_ARG;
    RenderPassColorAttachment color = {};
    color.view = m_hdr->getDefaultView();
    color.clearValue[0] = 0.001f;
    color.clearValue[1] = 0.002f;
    color.clearValue[2] = 0.005f;
    color.clearValue[3] = 1;
    RenderPassDesc desc = {};
    desc.colorAttachments = &color;
    desc.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(desc);
    uint32_t region = profiler ? profiler->beginRegion(pass, "Particles + logo") : GpuProfiler::kInvalidRegion;
    pass->pushDebugGroup("Particle HDR scene", {});
    float aspect = float(m_width) / m_height;
    auto state = renderState();
    state.vertexBuffers[0] = {m_logoVertices, 0};
    state.vertexBufferCount = 1;
    state.indexBuffer = {m_logoIndices, 0};
    state.indexFormat = IndexFormat::Uint32;
    pass->setRenderState(state);
    for (auto& mesh : logo::kMeshRanges)
    {
        ShaderCursor cursor(pass->bindPipeline(m_logoPipeline));
        SLANG_RETURN_ON_FAIL(cursor["aspect"].setData(aspect));
        math::float3 tint =
            mesh.materialIndex == 0 ? math::float3(1, 0.133f, 0.033f) : math::float3(0.014f, 0.515f, 0.584f);
        SLANG_RETURN_ON_FAIL(cursor["logoColor"].setData(tint));
        SLANG_RETURN_ON_FAIL(cursor["cycleTime"].setData(m_renderCycleTime));
        DrawArguments draw = {};
        draw.vertexCount = mesh.indexCount;
        draw.startIndexLocation = mesh.firstIndex;
        pass->drawIndexed(draw);
    }
    ShaderCursor cursor(pass->bindPipeline(m_particlePipeline));
    SLANG_RETURN_ON_FAIL(cursor["aspect"].setData(aspect));
    SLANG_RETURN_ON_FAIL(cursor["cycleTime"].setData(m_renderCycleTime));
    SLANG_RETURN_ON_FAIL(cursor["particles"].setBinding(m_particles[m_current]));
    SLANG_RETURN_ON_FAIL(cursor["liveCount"].setBinding(m_counts[m_current]));
    pass->setRenderState(renderState());
    if (m_indirect)
        pass->drawIndirect(1, {m_arguments, 0});
    else
    {
        DrawArguments draw = {};
        draw.vertexCount = 6;
        draw.instanceCount = m_capacity;
        pass->draw(draw);
    }
    if (profiler)
        profiler->endRegion(pass, region);
    pass->popDebugGroup();
    pass->end();

    if (bloom)
    {
        uint32_t bloomRegion = GpuProfiler::kInvalidRegion;
        for (uint32_t i = 0; i < 3; ++i)
        {
            auto compute = encoder->beginComputePass();
            if (profiler && i == 0)
                bloomRegion = profiler->beginRegion(compute, "Bloom");
            compute->pushDebugGroup("Particle bloom", {});
            ShaderCursor filter(compute->bindPipeline(m_bloomPipeline));
            SLANG_RETURN_ON_FAIL(filter["source"].setBinding(i == 0 ? m_hdr.get() : m_bloom[(i - 1) % 2].get()));
            SLANG_RETURN_ON_FAIL(filter["destination"].setBinding(m_bloom[i % 2]));
            SLANG_RETURN_ON_FAIL(filter["linearSampler"].setBinding(m_sampler));
            SLANG_RETURN_ON_FAIL(filter["bloomPass"].setData(i));
            compute->dispatchCompute(((m_width + 1) / 2 + 7) / 8, ((m_height + 1) / 2 + 7) / 8, 1);
            if (profiler && i == 2)
                profiler->endRegion(compute, bloomRegion);
            compute->popDebugGroup();
            compute->end();
        }
    }
    color.view = output->getDefaultView();
    pass = encoder->beginRenderPass(desc);
    region = profiler ? profiler->beginRegion(pass, "Tone map") : GpuProfiler::kInvalidRegion;
    cursor = ShaderCursor(pass->bindPipeline(m_toneMapPipeline));
    SLANG_RETURN_ON_FAIL(cursor["source"].setBinding(m_hdr));
    SLANG_RETURN_ON_FAIL(cursor["bloom"].setBinding(bloom ? m_bloom[0].get() : m_hdr.get()));
    SLANG_RETURN_ON_FAIL(cursor["linearSampler"].setBinding(m_sampler));
    SLANG_RETURN_ON_FAIL(cursor["exposure"].setData(0.0f));
    SLANG_RETURN_ON_FAIL(cursor["bloomStrength"].setData(bloom ? 0.35f : 0.0f));
    SLANG_RETURN_ON_FAIL(cursor["encodeSrgb"].setData(getFormatInfo(m_outputFormat).isSrgb ? 0u : 1u));
    pass->setRenderState(renderState());
    DrawArguments draw = {};
    draw.vertexCount = 3;
    pass->draw(draw);
    if (profiler)
        profiler->endRegion(pass, region);
    pass->end();
    return SLANG_OK;
}

IBuffer* Renderer::countBuffer() const
{
    return m_counts[m_current];
}

IBuffer* Renderer::particleBuffer() const
{
    return m_particles[m_current];
}

IBuffer* Renderer::argumentBuffer() const
{
    return m_arguments;
}

bool Renderer::indirect() const
{
    return m_indirect;
}

void Renderer::setIndirect(bool enabled)
{
    m_indirect = enabled && m_device->getDeviceType() != DeviceType::Metal;
}

uint32_t Renderer::capacity() const
{
    return m_capacity;
}

void Renderer::burst()
{
    m_cycleTick = PARTICLE_BURST_TICK;
}

const char* Renderer::phaseName() const
{
    if (m_renderCycleTime < 3)
        return "FLOW";
    if (m_renderCycleTime < 7)
        return "GATHER";
    if (m_renderCycleTime < 10)
        return "FORM";
    if (m_renderCycleTime < 11)
        return "BURST";
    return "SWIRL";
}

RenderState Renderer::renderState() const
{
    RenderState state = {};
    state.viewports[0] = Viewport::fromSize(m_width, m_height);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(m_width, m_height);
    state.scissorRectCount = 1;
    return state;
}

} // namespace rhi::particles

#ifndef SLANG_RHI_EXAMPLE_HEADLESS
#include "../base/example.h"

#include <memory>

using namespace rhi;

class ExampleParticles : public ExampleBase
{
public:
    ExampleUsage getUsage() const override
    {
        return {
            "Particle sculpture: GPU simulation and indirect drawing.",
            "Move mouse: Stir | Left: Attract | Right: Repel\n[ / ]: Emission | Space: Pause | N: Single step\nF: "
            "Burst | R: Reset | B: Bloom | D: Direct/indirect draw"
        };
    }

    Result init(DeviceType deviceType) override
    {
        SLANG_RETURN_ON_FAIL(
            createDevice(deviceType, {Feature::Surface, Feature::Rasterization}, {}, m_device.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createWindow(m_device, "Particle energy sculpture", 960, 720));
        SLANG_RETURN_ON_FAIL(createSurface(m_device, Format::Undefined, m_surface.writeRef()));
        SLANG_RETURN_ON_FAIL(m_device->getQueue(QueueType::Graphics, m_queue.writeRef()));
        particles::Programs programs;
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "particles.slang", {"simulateMain"}, programs.simulate.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createProgram(m_device, "particles.slang", {"spawnMain"}, programs.spawn.writeRef()));
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "particles.slang", {"finalizeMain"}, programs.finalize.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createProgram(
            m_device,
            "particles.slang",
            {"particleVertex", "particleFragment"},
            programs.particles.writeRef()
        ));
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "particles.slang", {"logoVertex", "logoFragment"}, programs.logo.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createProgram(m_device, "../raster/post.slang", {"bloomMain"}, programs.bloom.writeRef()));
        SLANG_RETURN_ON_FAIL(createProgram(
            m_device,
            "../raster/post.slang",
            {"fullscreenVertex", "toneMapFragment"},
            programs.toneMap.writeRef()
        ));
        m_renderer = std::make_unique<particles::Renderer>();
        SLANG_RETURN_ON_FAIL(m_renderer->init(m_device, m_surface->getConfig()->format, programs));
        SLANG_RETURN_ON_FAIL(m_renderer->resize(m_surface->getConfig()->width, m_surface->getConfig()->height));
        m_profiler = std::make_unique<GpuProfiler>();
        SLANG_RETURN_ON_FAIL(m_profiler->init(m_device));


        // Live-count telemetry is independent of simulation and profiling. D3D11
        // has no RHI fence implementation, so it simply omits this optional HUD value.
        if (deviceType != DeviceType::D3D11)
        {
            FenceDesc fence = {};
            SLANG_RETURN_ON_FAIL(m_device->createFence(fence, m_readbackFence.writeRef()));
            for (auto& slot : m_readbacks)
            {
                BufferDesc desc = {};
                desc.size = sizeof(uint32_t);
                desc.usage = BufferUsage::CopyDestination;
                desc.memoryType = MemoryType::ReadBack;
                SLANG_RETURN_ON_FAIL(m_device->createBuffer(desc, nullptr, slot.buffer.writeRef()));
            }
        }
        return SLANG_OK;
    }

    void shutdown() override
    {
        if (m_queue)
            m_queue->waitOnHost();
        m_renderer.reset();
        m_profiler.reset();
        m_readbacks = {};
        m_readbackFence.setNull();
        m_surface.setNull();
        m_queue.setNull();
        m_device.setNull();
    }

    Result update(double time) override
    {
        bool visible = m_surface->getConfig() != nullptr;
        if (!isMouseHovered())
        {
            m_mouseValid = false;
            m_mouseSimulatedWorld = m_mouseWorld;
        }
        m_steps = m_clock.update(time, visible && !m_paused);
        if (visible && m_singleStep && m_paused)
            m_steps = 1;
        m_singleStep = false;
        // Do not replay cursor travel collected while paused or minimized.
        if (!visible || (m_paused && !m_steps))
            m_mouseSimulatedWorld = m_mouseWorld;
        if (visible)
            m_frameStats.update(time);
        return m_resizeResult;
    }

    Result draw() override
    {
        if (!m_surface->getConfig())
            return SLANG_OK;
        ComPtr<ITexture> image;
        SLANG_RETURN_ON_FAIL(acquireFrame(m_surface, image.writeRef()));
        if (!image)
            return SLANG_OK;
        SLANG_RETURN_ON_FAIL(pollReadbacks());
        SLANG_RETURN_ON_FAIL(m_profiler->beginFrame());
        auto encoder = m_queue->createCommandEncoder();
        if (m_reset)
        {
            m_renderer->reset(encoder);
            m_reset = false;
            m_mouseSimulatedWorld = m_mouseWorld;
        }
        particles::SimulationParams params;
        params.spawnCount = m_emissionPerStep;
        params.mouse = m_mouseWorld;
        params.mouseForce = isMouseDown(GLFW_MOUSE_BUTTON_RIGHT)  ? -18.0f
                            : isMouseDown(GLFW_MOUSE_BUTTON_LEFT) ? 18.0f
                                                                  : 0.0f;
        // Distribute pending motion over the fixed steps, consuming it once.
        // Frames with no simulation step retain it for the next step.
        math::float2 travel = m_mouseWorld - m_mouseSimulatedWorld;
        if (m_mouseValid && m_steps)
        {
            params.mouseVelocity = travel / (float(m_steps) * params.dt);
            float speed = math::length(params.mouseVelocity);
            if (speed > 12)
                params.mouseVelocity = params.mouseVelocity * (12 / speed);
        }
        for (uint32_t i = 0; i < m_steps; ++i)
        {
            params.mousePrevious = m_mouseSimulatedWorld + travel * (float(i) / m_steps);
            params.mouse = m_mouseSimulatedWorld + travel * (float(i + 1) / m_steps);
            SLANG_RETURN_ON_FAIL(m_renderer->step(encoder, params, m_profiler.get()));
        }
        if (m_steps)
            m_mouseSimulatedWorld = m_mouseWorld;
        SLANG_RETURN_ON_FAIL(m_renderer->render(encoder, image, m_bloom, m_profiler.get()));
        SLANG_RETURN_ON_FAIL(drawOverlay(encoder, image));

        Readback* readback = nullptr;
        if (m_readbackFence && m_frameNumber++ % 30 == 0)
        {
            for (auto& slot : m_readbacks)
                if (!slot.value)
                {
                    readback = &slot;
                    encoder->copyBuffer(slot.buffer, 0, m_renderer->countBuffer(), 0, sizeof(uint32_t));
                    break;
                }
        }
        SLANG_RETURN_ON_FAIL(m_profiler->endFrame());
        ComPtr<ICommandBuffer> commands;
        SLANG_RETURN_ON_FAIL(encoder->finish(commands.writeRef()));
        ICommandBuffer* commandPointer = commands;
        IFence* fencePointer = m_readbackFence;
        SubmitDesc submit = {};
        submit.commandBuffers = &commandPointer;
        submit.commandBufferCount = 1;
        if (readback)
        {
            readback->value = ++m_fenceValue;
            readback->generation = m_generation;
            submit.signalFences = &fencePointer;
            submit.signalFenceValues = &readback->value;
            submit.signalFenceCount = 1;
        }
        SLANG_RETURN_ON_FAIL(m_queue->submit(submit));
        return presentFrame(m_surface, m_queue);
    }

    void onResize(int width, int height, int framebufferWidth, int framebufferHeight) override
    {
        if (!m_queue || !m_renderer)
            return;
        m_frameStats.reset();
        m_clock.reset();
        m_mouseValid = false;
        m_resizeResult = m_queue->waitOnHost();
        if (SLANG_FAILED(m_resizeResult))
            return;
        if (framebufferWidth <= 0 || framebufferHeight <= 0)
        {
            m_resizeResult = m_surface->unconfigure();
            return;
        }
        SurfaceConfig config = {};
        config.format = m_surface->getInfo().preferredFormat;
        config.width = framebufferWidth;
        config.height = framebufferHeight;
        m_resizeResult = m_surface->configure(config);
        if (SLANG_SUCCEEDED(m_resizeResult))
            m_resizeResult = m_renderer->resize(framebufferWidth, framebufferHeight);
    }

    void onMousePosition(float x, float y) override
    {
        // Any backend window can supply mirrored input. Normalize using the
        // actual source window, and never sweep between different windows.
        auto source = getMouseSourceWindow();
        int width, height;
        glfwGetWindowSize(source, &width, &height);
        if (width > 0 && height > 0)
        {
            m_mouseWorld = {(2 * x / width - 1) * 3.6f * float(width) / height, (1 - 2 * y / height) * 3.6f};
            bool inside = x >= 0 && y >= 0 && x < width && y < height;
            if (!m_mouseValid || !inside || source != m_lastMouseSource)
                m_mouseSimulatedWorld = m_mouseWorld;
            m_mouseValid = inside;
            m_lastMouseSource = source;
        }
    }

    void onKey(int key, int scancode, int action, int mods) override
    {
        if (action != GLFW_PRESS && action != GLFW_REPEAT)
            return;
        if (key == GLFW_KEY_LEFT_BRACKET)
            m_emissionPerStep = m_emissionPerStep > 100 ? m_emissionPerStep - 100 : 0;
        if (key == GLFW_KEY_RIGHT_BRACKET)
            m_emissionPerStep = std::min(4000u, m_emissionPerStep + 100);
        if (action != GLFW_PRESS)
            return;
        if (key == GLFW_KEY_SPACE)
        {
            m_paused = !m_paused;
            m_mouseSimulatedWorld = m_mouseWorld;
        }
        if (key == GLFW_KEY_N)
            m_singleStep = true;
        if (key == GLFW_KEY_B)
            m_bloom = !m_bloom;
        if (key == GLFW_KEY_D)
            m_renderer->setIndirect(!m_renderer->indirect());
        if (key == GLFW_KEY_F)
            m_renderer->burst();
        if (key == GLFW_KEY_R)
        {
            m_reset = true;
            m_clock.reset();
            m_countValid = false;
            ++m_generation;
        }
    }

private:
    struct Readback
    {
        ComPtr<IBuffer> buffer;
        uint64_t value = 0, generation = 0;
    };

    Result pollReadbacks()
    {
        if (!m_readbackFence)
            return SLANG_OK;
        uint64_t completed;
        SLANG_RETURN_ON_FAIL(m_readbackFence->getCurrentValue(&completed));
        for (auto& slot : m_readbacks)
        {
            if (!slot.value || slot.value > completed)
                continue;
            if (slot.generation == m_generation && slot.value > m_lastReadback)
            {
                void* data;
                SLANG_RETURN_ON_FAIL(m_device->mapBuffer(slot.buffer, CpuAccessMode::Read, &data));
                m_liveCount = *static_cast<const uint32_t*>(data);
                SLANG_RETURN_ON_FAIL(m_device->unmapBuffer(slot.buffer));
                m_countValid = true;
                m_lastReadback = slot.value;
            }
            slot.value = 0;
        }
        return SLANG_OK;
    }

    Result drawOverlay(ICommandEncoder* encoder, ITexture* image)
    {
        char text[1024], count[64];
        if (m_countValid)
            snprintf(count, sizeof(count), "%u / %u (DELAYED)", m_liveCount, m_renderer->capacity());
        else
            snprintf(count, sizeof(count), "%s", m_readbackFence ? "PENDING" : "READBACK UNAVAILABLE");
        double simulation = 0, rendering = 0, post = 0;
        for (const auto& sample : m_profiler->samples())
        {
            if (sample.name == "Simulation step")
                simulation += sample.milliseconds;
            else if (sample.name == "Particles + logo")
                rendering += sample.milliseconds;
            else
                post += sample.milliseconds;
        }
        char timing[160];
        if (!m_profiler->supported())
            snprintf(timing, sizeof(timing), "GPU TIMING UNAVAILABLE");
        else if (m_profiler->samples().empty())
            snprintf(timing, sizeof(timing), "GPU TIMING PENDING");
        else
            snprintf(timing, sizeof(timing), "GPU MS: SIM %.2f / DRAW %.2f / POST %.2f", simulation, rendering, post);
        snprintf(
            text,
            sizeof(text),
            "PARTICLE SCULPTURE / %s\n%.1f FPS / %.2f MS APP LOOP\n%s\nLIVE %s\n%s / EMIT %u PER SEC / %s\n"
            "PHASE: %s",
            getRHI()->getDeviceTypeName(m_deviceType),
            m_frameStats.fps(),
            m_frameStats.milliseconds(),
            timing,
            count,
            m_renderer->indirect() ? "INDIRECT" : "DIRECT CAPACITY",
            m_emissionPerStep * 120,
            m_paused ? "PAUSED" : "RUNNING",
            m_renderer->phaseName()
        );
        return drawUsage(encoder, image, text);
    }

    ComPtr<IDevice> m_device;
    ComPtr<ISurface> m_surface;
    ComPtr<ICommandQueue> m_queue;
    std::unique_ptr<particles::Renderer> m_renderer;
    std::unique_ptr<GpuProfiler> m_profiler;
    ComPtr<IFence> m_readbackFence;
    std::array<Readback, 3> m_readbacks;
    uint64_t m_fenceValue = 0, m_lastReadback = 0, m_generation = 0, m_frameNumber = 0;
    FixedStepClock m_clock;
    FrameStats m_frameStats;
    math::float2 m_mouseWorld = {}, m_mouseSimulatedWorld = {};
    bool m_mouseValid = false;
    GLFWwindow* m_lastMouseSource = nullptr;
    uint32_t m_steps = 0, m_emissionPerStep = 400, m_liveCount = 0;
    bool m_paused = false, m_singleStep = false, m_reset = true, m_bloom = true, m_countValid = false;
    Result m_resizeResult = SLANG_OK;
};

EXAMPLE_MAIN(ExampleParticles)

#endif
