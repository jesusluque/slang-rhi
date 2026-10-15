// Rasterization example: a Slang logo and instanced material spheres lit by a
// directional light and procedural environments. The frame records a shadow pass,
// an HDR scene with optional 4x MSAA, compute bloom, and fullscreen tone mapping.
// Shader cursors bind resources and parameters; debug groups label the GPU passes.
// Drag to orbit the camera or move the light, and scroll to zoom. The overlay lists
// the remaining controls; F1 toggles it. See ../README.md for build and run commands.

#include "../base/example.h"
#include "../base/logo-scene.h"
#include "../base/gpu-profiler.h"
#include "../base/camera.h"
#include "../base/geometry.h"

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include <algorithm>
#include <memory>

using namespace rhi;

// Startup-only preprocessing. Compute writes ordinary array textures, then copies
// the faces to sampled cubemaps, including on backends without cube UAV views.
class Environment
{
public:
    static constexpr uint32_t kSpecularSize = 256;
    static constexpr uint32_t kSpecularMips = 9;
    static constexpr uint32_t kDiffuseSize = 16;
    static constexpr uint32_t kBrdfSize = 128;

    Result init(IDevice* device, IShaderProgram* filterProgram, IShaderProgram* brdfProgram);

    ComPtr<ITexture> specular[2], diffuse[2], brdf;
};


// Three float4s give StructuredBuffer elements the same 48-byte layout on every
// backend. Uniform positive scale lets the unit-sphere normal remain unchanged.
struct SphereInstance
{
    math::float4 translationScale;
    math::float4 baseColorMetallic;
    math::float4 materialParameters; // x: roughness; yzw: unused.
};
static_assert(sizeof(SphereInstance) == 48);

inline constexpr uint32_t kSphereColumns = 6;
inline constexpr uint32_t kSphereRows = 5;
inline constexpr uint32_t kSphereCount = kSphereColumns * kSphereRows;

std::vector<SphereInstance> createSphereInstances();


struct Settings
{
    float yaw = 0.22f;
    float pitch = 0.5f;
    float distance = 14.0f;
    float lightYaw = -0.65f;
    float lightPitch = 0.85f;
    float roughness = 0.28f;
    float exposure = 0.0f;
    bool bloom = true;
    uint32_t environmentPreset = 0; // Studio / outdoor.
    float environmentRotation = 0;
    bool environmentLighting = true;
    bool msaa = true;
    bool spheres = true;
};

struct Programs
{
    ComPtr<IShaderProgram> shadow;
    ComPtr<IShaderProgram> scene;
    ComPtr<IShaderProgram> bloom;
    ComPtr<IShaderProgram> toneMap;
    ComPtr<IShaderProgram> background;
    ComPtr<IShaderProgram> environment;
    ComPtr<IShaderProgram> environmentBrdf;
    ComPtr<IShaderProgram> spheres;
    ComPtr<IShaderProgram> sphereShadow;
};

// Records the rendering passes into the supplied encoder and output texture.
class Renderer
{
public:
    static constexpr uint32_t kShadowSize = 1024;

    Result init(IDevice* device, Format outputFormat, const Programs& programs);

    // The caller must wait for previous submissions before replacing these targets.
    Result resize(uint32_t width, uint32_t height);

    Result render(
        ICommandEncoder* encoder,
        ITexture* output,
        const Settings& settings,
        GpuProfiler* profiler = nullptr
    );

    bool supportsMsaa() const;

private:
    Result initSpheres();

    Result createMsaaTargets(uint32_t width, uint32_t height, ITexture** color, ITexture** depth);

    Result initMsaa(const Programs& programs);

    Result createTexture(
        uint32_t width,
        uint32_t height,
        Format format,
        TextureUsage usage,
        const char* label,
        ITexture** out,
        uint32_t sampleCount = 1
    );

    RenderState renderState(uint32_t width, uint32_t height, bool spheres = false) const;

    ComPtr<IDevice> m_device;
    Format m_outputFormat = Format::Undefined;
    uint32_t m_width = 0, m_height = 0;
    logo::Scene m_scene;
    ComPtr<IBuffer> m_vertices, m_indices;
    ComPtr<IInputLayout> m_inputLayout, m_shadowInputLayout;
    ComPtr<IRenderPipeline> m_shadowPipeline, m_scenePipeline, m_toneMapPipeline;
    ComPtr<IRenderPipeline> m_backgroundPipeline;
    ComPtr<IRenderPipeline> m_msaaScenePipeline, m_msaaBackgroundPipeline;
    ComPtr<IComputePipeline> m_bloomPipeline;
    ComPtr<ISampler> m_linearSampler, m_shadowSampler;
    ComPtr<ITexture> m_shadow, m_hdr, m_depth, m_bloom[2];
    Environment m_environment;
    ComPtr<ITexture> m_msaaColor, m_msaaDepth;
    bool m_msaaSupported = false;
    ComPtr<IBuffer> m_sphereVertices, m_sphereIndices, m_sphereInstances;
    uint32_t m_sphereIndexCount = 0;
    ComPtr<IRenderPipeline> m_spherePipeline, m_msaaSpherePipeline, m_sphereShadowPipeline;
};

Result Environment::init(IDevice* device, IShaderProgram* filterProgram, IShaderProgram* brdfProgram)
{
    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = filterProgram;
    ComPtr<IComputePipeline> filterPipeline;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(pipelineDesc, filterPipeline.writeRef()));
    pipelineDesc.program = brdfProgram;
    ComPtr<IComputePipeline> brdfPipeline;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(pipelineDesc, brdfPipeline.writeRef()));

    TextureDesc desc = {};
    desc.format = Format::RGBA16Float;
    desc.size = {kBrdfSize, kBrdfSize, 1};
    desc.usage = TextureUsage::ShaderResource | TextureUsage::UnorderedAccess;
    desc.label = "Environment BRDF LUT";
    SLANG_RETURN_ON_FAIL(device->createTexture(desc, nullptr, brdf.writeRef()));

    ComPtr<ICommandQueue> queue;
    SLANG_RETURN_ON_FAIL(device->getQueue(QueueType::Graphics, queue.writeRef()));
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    pass->pushDebugGroup("Integrate environment BRDF", {});
    ShaderCursor cursor(pass->bindPipeline(brdfPipeline));
    SLANG_RETURN_ON_FAIL(cursor["brdfOutput"].setBinding(brdf));
    pass->dispatchCompute((kBrdfSize + 7) / 8, (kBrdfSize + 7) / 8, 1);
    pass->popDebugGroup();
    pass->end();

    // Keep temporary textures/views alive until the initialization submission finishes.
    ComPtr<ITexture> staging[2];
    ComPtr<ITextureView> views[2][kSpecularMips];
    for (uint32_t diffuseFilter = 0; diffuseFilter < 2; ++diffuseFilter)
    {
        uint32_t size = diffuseFilter ? kDiffuseSize : kSpecularSize;
        uint32_t mipCount = diffuseFilter ? 1 : kSpecularMips;
        desc.type = TextureType::Texture2DArray;
        desc.size = {size, size, 1};
        desc.arrayLength = 6;
        desc.mipCount = mipCount;
        desc.usage = TextureUsage::UnorderedAccess | TextureUsage::CopySource;
        desc.label = "Environment filter staging";
        SLANG_RETURN_ON_FAIL(device->createTexture(desc, nullptr, staging[diffuseFilter].writeRef()));
        for (uint32_t mip = 0; mip < mipCount; ++mip)
        {
            TextureViewDesc view = {};
            view.subresourceRange = {0, 6, mip, 1};
            SLANG_RETURN_ON_FAIL(
                device->createTextureView(staging[diffuseFilter], view, views[diffuseFilter][mip].writeRef())
            );
        }

        desc.type = TextureType::TextureCube;
        desc.arrayLength = 1;
        desc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
        for (uint32_t preset = 0; preset < 2; ++preset)
        {
            auto& texture = diffuseFilter ? diffuse[preset] : specular[preset];
            desc.label = diffuseFilter ? "Environment diffuse irradiance" : "Environment GGX reflection";
            SLANG_RETURN_ON_FAIL(device->createTexture(desc, nullptr, texture.writeRef()));
            for (uint32_t mip = 0; mip < mipCount; ++mip)
            {
                uint32_t mipSize = std::max(size >> mip, 1u);
                float roughness = diffuseFilter ? 0 : float(mip) / (kSpecularMips - 1);
                pass = encoder->beginComputePass();
                pass->pushDebugGroup(diffuseFilter ? "Filter diffuse environment" : "Filter GGX environment", {});
                cursor = ShaderCursor(pass->bindPipeline(filterPipeline));
                SLANG_RETURN_ON_FAIL(cursor["environmentOutput"].setBinding(views[diffuseFilter][mip]));
                SLANG_RETURN_ON_FAIL(cursor["environmentPreset"].setData(preset));
                SLANG_RETURN_ON_FAIL(cursor["diffuseFilter"].setData(diffuseFilter));
                SLANG_RETURN_ON_FAIL(cursor["filterRoughness"].setData(roughness));
                pass->dispatchCompute((mipSize + 7) / 8, (mipSize + 7) / 8, 6);
                pass->popDebugGroup();
                pass->end();
                encoder->copyTexture(
                    texture,
                    {0, 6, mip, 1},
                    {},
                    staging[diffuseFilter],
                    {0, 6, mip, 1},
                    {},
                    {mipSize, mipSize, 1}
                );
            }
        }
    }
    ComPtr<ICommandBuffer> commands;
    SLANG_RETURN_ON_FAIL(encoder->finish(commands.writeRef()));
    SLANG_RETURN_ON_FAIL(queue->submit(commands));
    return queue->waitOnHost();
}

std::vector<SphereInstance> createSphereInstances()
{
    std::vector<SphereInstance> instances;
    for (uint32_t row = 0; row < kSphereRows; ++row)
    {
        for (uint32_t column = 0; column < kSphereColumns; ++column)
        {
            float t = float(column) / (kSphereColumns - 1);
            float metallic = float(row) / (kSphereRows - 1);
            float x = column < 3 ? -4.6f + 1.05f * column : 2.5f + 1.05f * (column - 3);
            constexpr float radius = 0.38f;
            math::float3 color = math::float3(1.0f, 0.18f, 0.05f) * (1 - t) + math::float3(0.02f, 0.5f, 0.6f) * t;
            instances.push_back({
                {x, radius - 0.03f, 2.6f - 1.3f * row, radius},
                math::float4(color, metallic),
                {0.08f + 0.92f * t, 0, 0, 0},
            });
        }
    }
    return instances;
}


Result Renderer::init(IDevice* device, Format outputFormat, const Programs& programs)
{
    m_device = device;
    m_outputFormat = outputFormat;
    m_scene = logo::createScene();

    // The bloom shader declares rgba16f storage explicitly.
    for (const auto& requirement : {
             std::pair{
                 Format::RGBA16Float,
                 FormatSupport::RenderTarget | FormatSupport::ShaderSample | FormatSupport::ShaderUavStore
             },
             // D3D's format table queries the depth format's SRV mapping,
             // so depth attachment support is verified by texture creation.
             std::pair{Format::D32Float, FormatSupport::ShaderSample},
         })
    {
        FormatSupport support;
        SLANG_RETURN_ON_FAIL(device->getFormatSupport(requirement.first, &support));
        if ((support & requirement.second) != requirement.second)
            return SLANG_E_NOT_AVAILABLE;
    }

    BufferDesc buffer = {};
    buffer.size = m_scene.vertices.size() * sizeof(logo::Vertex);
    buffer.usage = BufferUsage::VertexBuffer;
    buffer.defaultState = ResourceState::VertexBuffer;
    buffer.label = "Logo vertices";
    SLANG_RETURN_ON_FAIL(device->createBuffer(buffer, m_scene.vertices.data(), m_vertices.writeRef()));
    buffer.size = m_scene.indices.size() * sizeof(uint32_t);
    buffer.usage = BufferUsage::IndexBuffer;
    buffer.defaultState = ResourceState::IndexBuffer;
    buffer.label = "Logo indices";
    SLANG_RETURN_ON_FAIL(device->createBuffer(buffer, m_scene.indices.data(), m_indices.writeRef()));
    SLANG_RETURN_ON_FAIL(initSpheres());

    VertexStreamDesc stream = {sizeof(logo::Vertex), InputSlotClass::PerVertex, 0};
    InputElementDesc elements[] = {
        {"POSITION", 0, Format::RGB32Float, offsetof(logo::Vertex, position), 0},
        {"NORMAL", 0, Format::RGB32Float, offsetof(logo::Vertex, normal), 0},
    };
    InputLayoutDesc input = {};
    input.vertexStreams = &stream;
    input.vertexStreamCount = 1;
    input.inputElements = elements;
    input.inputElementCount = 2;
    SLANG_RETURN_ON_FAIL(device->createInputLayout(input, m_inputLayout.writeRef()));
    input.inputElementCount = 1;
    SLANG_RETURN_ON_FAIL(device->createInputLayout(input, m_shadowInputLayout.writeRef()));

    RenderPipelineDesc pipeline = {};
    pipeline.inputLayout = m_shadowInputLayout;
    pipeline.depthStencil.format = Format::D32Float;
    pipeline.depthStencil.depthTestEnable = true;
    pipeline.depthStencil.depthWriteEnable = true;
    pipeline.program = programs.shadow;
    pipeline.rasterizer.depthBias = 100;
    pipeline.rasterizer.slopeScaledDepthBias = 1.5f;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_shadowPipeline.writeRef()));
    pipeline.program = programs.sphereShadow;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_sphereShadowPipeline.writeRef()));

    ColorTargetDesc color = {};
    color.format = Format::RGBA16Float;
    pipeline.targets = &color;
    pipeline.targetCount = 1;
    pipeline.inputLayout = m_inputLayout;
    pipeline.program = programs.scene;
    pipeline.rasterizer.depthBias = 0;
    pipeline.rasterizer.slopeScaledDepthBias = 0;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_scenePipeline.writeRef()));
    pipeline.program = programs.spheres;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_spherePipeline.writeRef()));

    pipeline = {};
    pipeline.targets = &color;
    pipeline.targetCount = 1;
    pipeline.depthStencil.depthWriteEnable = false;
    pipeline.program = programs.background;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_backgroundPipeline.writeRef()));

    // There is no per-format sample-count query in the RHI. Probe the exact
    // 4x color/depth attachments and pipelines, retaining the 1x path if unavailable.
    m_msaaSupported = SLANG_SUCCEEDED(initMsaa(programs));

    pipeline = {};
    color.format = outputFormat;
    pipeline.targets = &color;
    pipeline.targetCount = 1;
    pipeline.depthStencil.depthWriteEnable = false;
    pipeline.program = programs.toneMap;
    SLANG_RETURN_ON_FAIL(device->createRenderPipeline(pipeline, m_toneMapPipeline.writeRef()));

    ComputePipelineDesc compute = {};
    compute.program = programs.bloom;
    SLANG_RETURN_ON_FAIL(device->createComputePipeline(compute, m_bloomPipeline.writeRef()));

    SamplerDesc sampler = {};
    sampler.addressU = sampler.addressV = sampler.addressW = TextureAddressingMode::ClampToEdge;
    SLANG_RETURN_ON_FAIL(device->createSampler(sampler, m_linearSampler.writeRef()));
    sampler.reductionOp = TextureReductionOp::Comparison;
    sampler.comparisonFunc = ComparisonFunc::LessEqual;
    SLANG_RETURN_ON_FAIL(device->createSampler(sampler, m_shadowSampler.writeRef()));
    SLANG_RETURN_ON_FAIL(m_environment.init(device, programs.environment, programs.environmentBrdf));
    return createTexture(
        kShadowSize,
        kShadowSize,
        Format::D32Float,
        TextureUsage::DepthStencil | TextureUsage::ShaderResource,
        "Shadow map",
        m_shadow.writeRef()
    );
}

Result Renderer::resize(uint32_t width, uint32_t height)
{
    if (!width || !height)
        return SLANG_E_INVALID_ARG;
    m_width = width;
    m_height = height;
    SLANG_RETURN_ON_FAIL(createTexture(
        width,
        height,
        Format::RGBA16Float,
        TextureUsage::RenderTarget | TextureUsage::ShaderResource |
            (m_msaaSupported ? TextureUsage::ResolveDestination : TextureUsage::None),
        "HDR scene",
        m_hdr.writeRef()
    ));
    SLANG_RETURN_ON_FAIL(
        createTexture(width, height, Format::D32Float, TextureUsage::DepthStencil, "Scene depth", m_depth.writeRef())
    );
    if (m_msaaSupported)
        SLANG_RETURN_ON_FAIL(createMsaaTargets(width, height, m_msaaColor.writeRef(), m_msaaDepth.writeRef()));
    for (auto& target : m_bloom)
        SLANG_RETURN_ON_FAIL(createTexture(
            (width + 1) / 2,
            (height + 1) / 2,
            Format::RGBA16Float,
            TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
            "Bloom",
            target.writeRef()
        ));
    return SLANG_OK;
}

Result Renderer::render(ICommandEncoder* encoder, ITexture* output, const Settings& settings, GpuProfiler* profiler)
{
    if (!m_width || !m_height || output->getDesc().size.width != m_width || output->getDesc().size.height != m_height ||
        output->getDesc().format != m_outputFormat || settings.environmentPreset > 1)
        return SLANG_E_INVALID_ARG;
    bool msaa = settings.msaa && m_msaaSupported;
    math::float3 target(0, 1.58f, 0);
    math::float3 eye = target + orbitDirection(settings.yaw, settings.pitch) * settings.distance;
    math::float3 light = orbitDirection(settings.lightYaw, settings.lightPitch);
    math::float4x4 viewProjection =
        math::mul(math::perspective(float(m_width) / m_height, 0.65f, 0.1f, 60.0f), math::lookAt(eye, target));
    math::float4x4 orthographic = math::identityMatrix<4>();
    orthographic[0][0] = orthographic[1][1] = 1.0f / 7.0f;
    orthographic[2][2] = 1.0f / (0.1f - 24.0f);
    orthographic[2][3] = 0.1f / (0.1f - 24.0f);
    math::float4x4 lightViewProjection = math::mul(orthographic, math::lookAt(target + light * 12.0f, target));

    RenderPassDepthStencilAttachment depth = {};
    depth.view = m_shadow->getDefaultView();
    RenderPassDesc pass = {};
    pass.depthStencilAttachment = &depth;
    auto renderPass = encoder->beginRenderPass(pass);
    uint32_t profileRegion = profiler ? profiler->beginRegion(renderPass, "Raster frame") : GpuProfiler::kInvalidRegion;
    renderPass->pushDebugGroup("Shadow map", {});
    ShaderCursor shadowCursor(renderPass->bindPipeline(m_shadowPipeline));
    SLANG_RETURN_ON_FAIL(shadowCursor["frame"]["lightViewProjection"].setData(lightViewProjection));
    renderPass->setRenderState(renderState(kShadowSize, kShadowSize));
    DrawArguments draw = {};
    draw.vertexCount = uint32_t(m_scene.indices.size());
    renderPass->drawIndexed(draw);
    if (settings.spheres)
    {
        ShaderCursor cursor(renderPass->bindPipeline(m_sphereShadowPipeline));
        SLANG_RETURN_ON_FAIL(cursor["frame"]["lightViewProjection"].setData(lightViewProjection));
        SLANG_RETURN_ON_FAIL(cursor["sphereInstances"].setBinding(m_sphereInstances));
        renderPass->setRenderState(renderState(kShadowSize, kShadowSize, true));
        DrawArguments sphereDraw = {};
        sphereDraw.vertexCount = m_sphereIndexCount;
        sphereDraw.instanceCount = kSphereCount;
        renderPass->drawIndexed(sphereDraw);
    }
    renderPass->popDebugGroup();
    renderPass->end();

    RenderPassColorAttachment color = {};
    color.view = (msaa ? m_msaaColor : m_hdr)->getDefaultView();
    color.clearValue[0] = 0.018f;
    color.clearValue[1] = 0.025f;
    color.clearValue[2] = 0.04f;
    color.clearValue[3] = 1.0f;
    depth.view = (msaa ? m_msaaDepth : m_depth)->getDefaultView();
    pass.colorAttachments = &color;
    pass.colorAttachmentCount = 1;
    pass.depthStencilAttachment = nullptr;
    renderPass = encoder->beginRenderPass(pass);
    renderPass->pushDebugGroup("Procedural environment background", {});
    ShaderCursor background(renderPass->bindPipeline(msaa ? m_msaaBackgroundPipeline : m_backgroundPipeline));
    math::float3 forward = math::normalize(target - eye);
    math::float3 right = math::normalize(math::cross(forward, math::float3(0, 1, 0)));
    float scale = std::tan(0.65f * 0.5f);
    math::float3 cameraRight = right * (scale * float(m_width) / m_height);
    math::float3 cameraUp = math::cross(right, forward) * scale;
    SLANG_RETURN_ON_FAIL(background["frame"]["cameraRight"].setData(cameraRight));
    SLANG_RETURN_ON_FAIL(background["frame"]["cameraUp"].setData(cameraUp));
    SLANG_RETURN_ON_FAIL(background["frame"]["cameraForward"].setData(forward));
    SLANG_RETURN_ON_FAIL(background["frame"]["lightDirection"].setData(light));
    SLANG_RETURN_ON_FAIL(background["frame"]["environmentRotation"].setData(settings.environmentRotation));
    SLANG_RETURN_ON_FAIL(background["frame"]["environmentPreset"].setData(settings.environmentPreset));
    RenderState backgroundState = {};
    backgroundState.viewports[0] = Viewport::fromSize(m_width, m_height);
    backgroundState.viewportCount = 1;
    backgroundState.scissorRects[0] = ScissorRect::fromSize(m_width, m_height);
    backgroundState.scissorRectCount = 1;
    renderPass->setRenderState(backgroundState);
    DrawArguments backgroundDraw = {};
    backgroundDraw.vertexCount = 3;
    renderPass->draw(backgroundDraw);
    renderPass->popDebugGroup();
    renderPass->end();

    color.loadOp = LoadOp::Load;
    // Preserve the multisampled background until geometry is complete, then
    // resolve linear HDR color before the single-sample post-processing passes.
    color.resolveTarget = msaa ? m_hdr->getDefaultView() : nullptr;
    pass.depthStencilAttachment = &depth;
    renderPass = encoder->beginRenderPass(pass);
    renderPass->pushDebugGroup("Lit HDR scene", {});
    renderPass->setRenderState(renderState(m_width, m_height));
    auto bindSceneFrame = [&](ShaderCursor cursor) -> Result
    {
        SLANG_RETURN_ON_FAIL(cursor["frame"]["viewProjection"].setData(viewProjection));
        SLANG_RETURN_ON_FAIL(cursor["frame"]["lightViewProjection"].setData(lightViewProjection));
        SLANG_RETURN_ON_FAIL(cursor["frame"]["eye"].setData(eye));
        SLANG_RETURN_ON_FAIL(cursor["frame"]["lightDirection"].setData(light));
        SLANG_RETURN_ON_FAIL(cursor["frame"]["environmentRotation"].setData(settings.environmentRotation));
        float environmentIntensity = settings.environmentLighting ? 1.0f : 0.0f;
        SLANG_RETURN_ON_FAIL(cursor["frame"]["environmentIntensity"].setData(environmentIntensity));
        SLANG_RETURN_ON_FAIL(
            cursor["specularEnvironment"].setBinding(m_environment.specular[settings.environmentPreset])
        );
        SLANG_RETURN_ON_FAIL(
            cursor["diffuseEnvironment"].setBinding(m_environment.diffuse[settings.environmentPreset])
        );
        SLANG_RETURN_ON_FAIL(cursor["environmentBrdf"].setBinding(m_environment.brdf));
        SLANG_RETURN_ON_FAIL(cursor["environmentSampler"].setBinding(m_linearSampler));
        SLANG_RETURN_ON_FAIL(cursor["shadowMap"].setBinding(m_shadow));
        SLANG_RETURN_ON_FAIL(cursor["shadowSampler"].setBinding(m_shadowSampler));
        return SLANG_OK;
    };
    for (const auto& mesh : m_scene.meshes)
    {
        ShaderCursor cursor(renderPass->bindPipeline(msaa ? m_msaaScenePipeline : m_scenePipeline));
        SLANG_RETURN_ON_FAIL(bindSceneFrame(cursor));
        const auto& material = m_scene.materials[mesh.materialIndex];
        float roughness = mesh.materialIndex < 2 ? settings.roughness : material.roughness;
        SLANG_RETURN_ON_FAIL(cursor["material"]["baseColor"].setData(material.baseColor));
        SLANG_RETURN_ON_FAIL(cursor["material"]["metallic"].setData(material.metallic));
        SLANG_RETURN_ON_FAIL(cursor["material"]["roughness"].setData(roughness));
        uint32_t groundPattern = mesh.materialIndex == 2 ? 1 : 0;
        SLANG_RETURN_ON_FAIL(cursor["material"]["groundPattern"].setData(groundPattern));
        draw.startIndexLocation = mesh.firstIndex;
        draw.vertexCount = mesh.indexCount;
        renderPass->drawIndexed(draw);
    }
    if (settings.spheres)
    {
        renderPass->pushDebugGroup("Instanced spheres", {});
        ShaderCursor cursor(renderPass->bindPipeline(msaa ? m_msaaSpherePipeline : m_spherePipeline));
        SLANG_RETURN_ON_FAIL(bindSceneFrame(cursor));
        SLANG_RETURN_ON_FAIL(cursor["sphereInstances"].setBinding(m_sphereInstances));
        renderPass->setRenderState(renderState(m_width, m_height, true));
        DrawArguments sphereDraw = {};
        sphereDraw.vertexCount = m_sphereIndexCount;
        sphereDraw.instanceCount = kSphereCount;
        renderPass->drawIndexed(sphereDraw);
        renderPass->popDebugGroup();
    }
    renderPass->popDebugGroup();
    renderPass->end();

    if (settings.bloom)
    {
        for (uint32_t i = 0; i < 3; ++i)
        {
            auto computePass = encoder->beginComputePass();
            computePass->pushDebugGroup(
                i == 0   ? "Bloom highlights"
                : i == 1 ? "Bloom horizontal"
                         : "Bloom vertical",
                {}
            );
            ShaderCursor cursor(computePass->bindPipeline(m_bloomPipeline));
            SLANG_RETURN_ON_FAIL(cursor["source"].setBinding(i == 0 ? m_hdr.get() : m_bloom[(i - 1) % 2].get()));
            SLANG_RETURN_ON_FAIL(cursor["destination"].setBinding(m_bloom[i % 2]));
            SLANG_RETURN_ON_FAIL(cursor["linearSampler"].setBinding(m_linearSampler));
            SLANG_RETURN_ON_FAIL(cursor["bloomPass"].setData(i));
            computePass->dispatchCompute(((m_width + 1) / 2 + 7) / 8, ((m_height + 1) / 2 + 7) / 8, 1);
            computePass->popDebugGroup();
            computePass->end();
        }
    }

    color.view = output->getDefaultView();
    color.resolveTarget = nullptr;
    color.loadOp = LoadOp::Clear;
    pass.depthStencilAttachment = nullptr;
    renderPass = encoder->beginRenderPass(pass);
    renderPass->pushDebugGroup("Tone mapping", {});
    ShaderCursor cursor(renderPass->bindPipeline(m_toneMapPipeline));
    SLANG_RETURN_ON_FAIL(cursor["source"].setBinding(m_hdr));
    SLANG_RETURN_ON_FAIL(cursor["bloom"].setBinding(settings.bloom ? m_bloom[0].get() : m_hdr.get()));
    SLANG_RETURN_ON_FAIL(cursor["linearSampler"].setBinding(m_linearSampler));
    SLANG_RETURN_ON_FAIL(cursor["exposure"].setData(settings.exposure));
    float strength = settings.bloom ? 0.18f : 0.0f;
    SLANG_RETURN_ON_FAIL(cursor["bloomStrength"].setData(strength));
    uint32_t encodeSrgb = getFormatInfo(m_outputFormat).isSrgb ? 0 : 1;
    SLANG_RETURN_ON_FAIL(cursor["encodeSrgb"].setData(encodeSrgb));
    RenderState state = {};
    state.viewports[0] = Viewport::fromSize(m_width, m_height);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(m_width, m_height);
    state.scissorRectCount = 1;
    renderPass->setRenderState(state);
    draw = {};
    draw.vertexCount = 3;
    renderPass->draw(draw);
    if (profiler)
        profiler->endRegion(renderPass, profileRegion);
    renderPass->popDebugGroup();
    renderPass->end();
    return SLANG_OK;
}

bool Renderer::supportsMsaa() const
{
    return m_msaaSupported;
}

Result Renderer::initSpheres()
{
    // The logo and sphere meshes share the position/normal input layout.
    auto mesh = geometry::Mesh::createSphere();
    m_sphereIndexCount = uint32_t(mesh.indices.size());
    BufferDesc desc = {};
    desc.size = mesh.vertices.size() * sizeof(geometry::Vertex);
    desc.usage = BufferUsage::VertexBuffer;
    desc.defaultState = ResourceState::VertexBuffer;
    desc.label = "Shared sphere vertices";
    SLANG_RETURN_ON_FAIL(m_device->createBuffer(desc, mesh.vertices.data(), m_sphereVertices.writeRef()));
    desc.size = mesh.indices.size() * sizeof(uint32_t);
    desc.usage = BufferUsage::IndexBuffer;
    desc.defaultState = ResourceState::IndexBuffer;
    desc.label = "Shared sphere indices";
    SLANG_RETURN_ON_FAIL(m_device->createBuffer(desc, mesh.indices.data(), m_sphereIndices.writeRef()));
    auto instances = createSphereInstances();
    desc.size = instances.size() * sizeof(SphereInstance);
    desc.elementSize = sizeof(SphereInstance);
    desc.usage = BufferUsage::ShaderResource;
    desc.defaultState = ResourceState::ShaderResource;
    desc.label = "Sphere transforms and materials";
    return m_device->createBuffer(desc, instances.data(), m_sphereInstances.writeRef());
}

Result Renderer::createMsaaTargets(uint32_t width, uint32_t height, ITexture** color, ITexture** depth)
{
    SLANG_RETURN_ON_FAIL(createTexture(
        width,
        height,
        Format::RGBA16Float,
        TextureUsage::RenderTarget | TextureUsage::ResolveSource,
        "4x MSAA HDR scene",
        color,
        4
    ));
    return createTexture(width, height, Format::D32Float, TextureUsage::DepthStencil, "4x MSAA depth", depth, 4);
}

Result Renderer::initMsaa(const Programs& programs)
{
    FormatSupport support = FormatSupport::None;
    SLANG_RETURN_ON_FAIL(m_device->getFormatSupport(Format::RGBA16Float, &support));
    if ((support & FormatSupport::Resolvable) == FormatSupport::None)
        return SLANG_E_NOT_AVAILABLE;
    ComPtr<ITexture> colorProbe, depthProbe;
    SLANG_RETURN_ON_FAIL(createMsaaTargets(1, 1, colorProbe.writeRef(), depthProbe.writeRef()));
    ColorTargetDesc color = {};
    color.format = Format::RGBA16Float;
    RenderPipelineDesc pipeline = {};
    pipeline.targets = &color;
    pipeline.targetCount = 1;
    pipeline.multisample.sampleCount = 4;
    pipeline.rasterizer.multisampleEnable = true;
    pipeline.depthStencil.depthWriteEnable = false;
    pipeline.program = programs.background;
    SLANG_RETURN_ON_FAIL(m_device->createRenderPipeline(pipeline, m_msaaBackgroundPipeline.writeRef()));
    pipeline.inputLayout = m_inputLayout;
    pipeline.depthStencil.format = Format::D32Float;
    pipeline.depthStencil.depthTestEnable = true;
    pipeline.depthStencil.depthWriteEnable = true;
    pipeline.program = programs.scene;
    SLANG_RETURN_ON_FAIL(m_device->createRenderPipeline(pipeline, m_msaaScenePipeline.writeRef()));
    pipeline.program = programs.spheres;
    return m_device->createRenderPipeline(pipeline, m_msaaSpherePipeline.writeRef());
}

Result Renderer::createTexture(
    uint32_t width,
    uint32_t height,
    Format format,
    TextureUsage usage,
    const char* label,
    ITexture** out,
    uint32_t sampleCount
)
{
    TextureDesc desc = {};
    desc.size = {width, height, 1};
    desc.format = format;
    desc.usage = usage;
    desc.label = label;
    desc.type = sampleCount > 1 ? TextureType::Texture2DMS : TextureType::Texture2D;
    desc.sampleCount = sampleCount;
    return m_device->createTexture(desc, nullptr, out);
}

RenderState Renderer::renderState(uint32_t width, uint32_t height, bool spheres) const
{
    RenderState state = {};
    state.viewports[0] = Viewport::fromSize(width, height);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(width, height);
    state.scissorRectCount = 1;
    state.vertexBuffers[0] = {spheres ? m_sphereVertices : m_vertices, 0};
    state.vertexBufferCount = 1;
    state.indexBuffer = {spheres ? m_sphereIndices : m_indices, 0};
    state.indexFormat = IndexFormat::Uint32;
    return state;
}

class ExampleRaster : public ExampleBase
{
public:
    Result init(DeviceType deviceType) override
    {
        SLANG_RETURN_ON_FAIL(
            createDevice(deviceType, {Feature::Surface, Feature::Rasterization}, {}, m_device.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createWindow(m_device, "Slang logo", 960, 640));
        SLANG_RETURN_ON_FAIL(createSurface(m_device, Format::Undefined, m_surface.writeRef()));
        SLANG_RETURN_ON_FAIL(initOverlay(m_device, m_surface->getConfig()->format));
        SLANG_RETURN_ON_FAIL(m_device->getQueue(QueueType::Graphics, m_queue.writeRef()));

        Programs programs;
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "raster.slang", {"shadowVertex", "shadowFragment"}, programs.shadow.writeRef())
        );
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "raster.slang", {"sceneVertex", "sceneFragment"}, programs.scene.writeRef())
        );
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "raster.slang", {"sphereVertex", "sceneFragment"}, programs.spheres.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createProgram(
            m_device,
            "raster.slang",
            {"sphereShadowVertex", "shadowFragment"},
            programs.sphereShadow.writeRef()
        ));
        SLANG_RETURN_ON_FAIL(createProgram(m_device, "post.slang", {"bloomMain"}, programs.bloom.writeRef()));
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "post.slang", {"fullscreenVertex", "toneMapFragment"}, programs.toneMap.writeRef())
        );
        SLANG_RETURN_ON_FAIL(createProgram(
            m_device,
            "raster.slang",
            {"backgroundVertex", "backgroundFragment"},
            programs.background.writeRef()
        ));
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "environment.slang", {"environmentMain"}, programs.environment.writeRef())
        );
        SLANG_RETURN_ON_FAIL(
            createProgram(m_device, "environment.slang", {"brdfMain"}, programs.environmentBrdf.writeRef())
        );
        m_renderer = std::make_unique<Renderer>();
        SLANG_RETURN_ON_FAIL(m_renderer->init(m_device, m_surface->getConfig()->format, programs));
        SLANG_RETURN_ON_FAIL(m_renderer->resize(m_surface->getConfig()->width, m_surface->getConfig()->height));

        m_profiler = std::make_unique<GpuProfiler>();
        SLANG_RETURN_ON_FAIL(m_profiler->init(m_device));
        updateTitle();
        return SLANG_OK;
    }

    void shutdown() override
    {
        if (m_queue)
            m_queue->waitOnHost();
        m_profiler.reset();
        m_renderer.reset();
        m_surface.setNull();
        m_queue.setNull();
        m_device.setNull();
    }

    Result update(double time) override
    {
        if (m_surface->getConfig() && m_frameStats.update(time))
            updateTitle();
        return m_resizeResult;
    }

    Result draw() override
    {
        if (!m_surface->getConfig())
            return SLANG_OK;
        ComPtr<ITexture> image;
        // A native resize can invalidate the swapchain before its queued callback.
        m_surface->acquireNextImage(image.writeRef());
        if (!image)
            return SLANG_OK;
        auto encoder = m_queue->createCommandEncoder();
        SLANG_RETURN_ON_FAIL(m_profiler->beginFrame());
        SLANG_RETURN_ON_FAIL(m_renderer->render(encoder, image, m_settings, m_profiler.get()));
        SLANG_RETURN_ON_FAIL(drawStatusOverlay(encoder, image));
        SLANG_RETURN_ON_FAIL(m_profiler->endFrame());
        ComPtr<ICommandBuffer> commands;
        SLANG_RETURN_ON_FAIL(encoder->finish(commands.writeRef()));
        SLANG_RETURN_ON_FAIL(m_queue->submit(commands));
        return m_surface->present();
    }

    void onResize(int width, int height, int framebufferWidth, int framebufferHeight) override
    {
        if (!m_queue || !m_renderer)
            return;
        m_frameStats.reset();
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

    void onMouseButton(int button, int action, int mods) override { m_lastMouse = {getMouseX(), getMouseY()}; }

    void onMousePosition(float x, float y) override
    {
        // The framework mirrors absolute positions to every backend window.
        // Use a fixed sensitivity so those shared events produce identical views.
        math::float2 delta = math::float2(x, y) - m_lastMouse;
        m_lastMouse = {x, y};
        if (isMouseDown(GLFW_MOUSE_BUTTON_LEFT))
        {
            m_cameraController.rotate(delta, m_settings.yaw, m_settings.pitch);
        }
        if (isMouseDown(GLFW_MOUSE_BUTTON_RIGHT))
        {
            m_settings.lightYaw -= delta.x * 0.006f;
            m_settings.lightPitch = std::clamp(m_settings.lightPitch + delta.y * 0.006f, 0.15f, 1.45f);
        }
    }

    void onScroll(float x, float y) override { m_cameraController.zoom(y, m_settings.distance); }

    void onKey(int key, int scancode, int action, int mods) override
    {
        if (action != GLFW_PRESS && action != GLFW_REPEAT)
            return;
        if (key == GLFW_KEY_B && action == GLFW_PRESS)
            m_settings.bloom = !m_settings.bloom;
        if (key == GLFW_KEY_E && action == GLFW_PRESS)
            m_settings.environmentPreset = 1 - m_settings.environmentPreset;
        if (key == GLFW_KEY_I && action == GLFW_PRESS)
            m_settings.environmentLighting = !m_settings.environmentLighting;
        if (key == GLFW_KEY_M && action == GLFW_PRESS && m_renderer->supportsMsaa())
            m_settings.msaa = !m_settings.msaa;
        if (key == GLFW_KEY_S && action == GLFW_PRESS)
            m_settings.spheres = !m_settings.spheres;
        if (key == GLFW_KEY_COMMA || key == GLFW_KEY_PERIOD)
            m_settings.environmentRotation =
                std::remainder(m_settings.environmentRotation + (key == GLFW_KEY_PERIOD ? 0.1f : -0.1f), 6.2831853f);
        if (key == GLFW_KEY_R)
            m_settings = {};
        if (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET)
            m_settings.roughness =
                std::clamp(m_settings.roughness + (key == GLFW_KEY_RIGHT_BRACKET ? 0.05f : -0.05f), 0.08f, 1.0f);
        if (key == GLFW_KEY_MINUS || key == GLFW_KEY_EQUAL)
            m_settings.exposure =
                std::clamp(m_settings.exposure + (key == GLFW_KEY_EQUAL ? 0.25f : -0.25f), -4.0f, 4.0f);
        updateTitle();
    }

private:
    const char* msaaStatus() const
    {
        return !m_renderer->supportsMsaa() ? "UNAVAILABLE" : m_settings.msaa ? "4X" : "OFF";
    }

    Result drawStatusOverlay(ICommandEncoder* encoder, ITexture* image)
    {
        char status[768], timing[96];
        if (!m_profiler->supported())
            snprintf(timing, sizeof(timing), "GPU timing unavailable");
        else if (m_profiler->samples().empty())
            snprintf(timing, sizeof(timing), "GPU timing pending");
        else
            snprintf(timing, sizeof(timing), "GPU render %.2f ms", m_profiler->samples()[0].milliseconds);
        snprintf(
            status,
            sizeof(status),
            "%s | %.1f FPS / %.2f ms app loop\n%s\n"
            "Roughness %.2f | Exposure %+.2f | Bloom %s\n"
            "MSAA %s | Spheres %u | Environment %s\n"
            "Environment angle %+.0f deg | Lighting %s",
            getRHI()->getDeviceTypeName(m_deviceType),
            m_frameStats.fps(),
            m_frameStats.milliseconds(),
            timing,
            m_settings.roughness,
            m_settings.exposure,
            m_settings.bloom ? "on" : "off",
            msaaStatus(),
            m_settings.spheres ? kSphereCount : 0u,
            m_settings.environmentPreset == 0 ? "studio" : "outdoor",
            m_settings.environmentRotation * (180.0f / 3.14159265f),
            m_settings.environmentLighting ? "on" : "off"
        );
        clearOverlay();
        printOverlay("Slang logo: rasterization, shadows and environment lighting.\n");
        printOverlay("Left drag: Orbit | Right drag: Light | Wheel: Zoom\n");
        printOverlay("[ / ]: Roughness | - / =: Exposure | B: Bloom\n");
        printOverlay("E: Environment | , / .: Rotate environment | I: Lighting\n");
        printOverlay("M: MSAA | S: Spheres | R: Reset\n");
        printOverlay("F1: Show/hide overlay | F2: Fit windows | Esc: Close\n\n%s", status);
        return drawOverlay(encoder, image);
    }

    void updateTitle()
    {
        char title[512];
        snprintf(
            title,
            sizeof(title),
            "Slang logo | %s | %.1f FPS / %.2f ms | roughness %.2f | exposure %+.2f | bloom %s | MSAA %s | %s | IBL %s "
            "| spheres %u",
            getRHI()->getDeviceTypeName(m_deviceType),
            m_frameStats.fps(),
            m_frameStats.milliseconds(),
            m_settings.roughness,
            m_settings.exposure,
            m_settings.bloom ? "on" : "off",
            msaaStatus(),
            m_settings.environmentPreset == 0 ? "studio" : "outdoor",
            m_settings.environmentLighting ? "on" : "off",
            m_settings.spheres ? kSphereCount : 0u
        );
        setWindowTitle(title);
    }

    ComPtr<IDevice> m_device;
    ComPtr<ISurface> m_surface;
    ComPtr<ICommandQueue> m_queue;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<GpuProfiler> m_profiler;
    FrameStats m_frameStats;
    Settings m_settings;
    math::float2 m_lastMouse;
    OrbitCameraController m_cameraController;
    Result m_resizeResult = SLANG_OK;
};

EXAMPLE_MAIN(ExampleRaster)
