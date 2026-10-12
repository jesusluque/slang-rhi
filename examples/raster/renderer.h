#pragma once

#include "../base/logo-scene.h"
#include "../base/gpu-profiler.h"
#include "../base/camera.h"
#include "../base/geometry.h"

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include <algorithm>

namespace rhi::raster {

using namespace math;

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

// The renderer records into a caller-owned encoder and output texture. Keeping
// window management outside it also lets tests render the complete scene offscreen.
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

} // namespace rhi::raster
