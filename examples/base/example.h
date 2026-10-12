#pragma once

#include "utils.h"
#include "text-renderer.h"
#include <memory>
#include <string_view>

#include <slang.h>
#include <slang-rhi.h>

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#if SLANG_WASM
#include <emscripten.h>
#include <emscripten/html5.h>
#else
#if SLANG_WINDOWS_FAMILY
#define GLFW_EXPOSE_NATIVE_WIN32
#elif SLANG_LINUX_FAMILY
#define GLFW_EXPOSE_NATIVE_X11
#elif SLANG_APPLE_FAMILY
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3native.h>
#include <slang-rhi/glfw.h>
#endif

namespace rhi {

class ExampleOverlay;

// Base class for examples.
// Derive from this class to implement an example.
class ExampleBase
{
public:
    ExampleBase();
    virtual ~ExampleBase();

    virtual ExampleUsage getUsage() const { return {}; }
    bool isUIVisible() const { return m_uiVisible; }
    void toggleUI() { m_uiVisible = !m_uiVisible; }

    // Draw after the example has finished rendering, before submission.
    // Scale multiplies the standard text size, independently of the window's pixel density.
    Result drawUsage(ICommandEncoder* encoder, ITexture* image, std::string_view status = {}, float scale = 1.5f);

    // Browser rendering uses an offscreen image: async buffer mapping can yield,
    // whereas a canvas image is only valid during the current browser task.
    Result acquireFrame(ISurface* surface, ITexture** image);
    Result presentFrame(ISurface* surface, ICommandQueue* queue);
    bool isMouseHovered() const;

    // Called to initialize the example for the specified device type.
    virtual Result init(DeviceType deviceType) = 0;
    // Called to shut down the example.
    virtual void shutdown() = 0;
    // Called every frame to update the example.
    virtual Result update(double time) = 0;
    // Called every frame to render the example.
    virtual Result draw() = 0;

    // Event callbacks

    // Called when the window is resized.
    virtual void onResize(int width, int height, int framebufferWidth, int framebufferHeight) {}
    // Called when the mouse is moved.
    virtual void onMousePosition(float x, float y) {}
    // Called when a mouse button is pressed or released.
    virtual void onMouseButton(int button, int action, int mods) {}
    // Called when the mouse wheel is scrolled.
    virtual void onScroll(float x, float y) {}
    // Called when a key is pressed, released, or repeated.
    virtual void onKey(int key, int scancode, int action, int mods) {}

    // Accessors for mouse state

    // Returns true if the specified mouse button is currently down.
    bool isMouseDown(int button) const
    {
        return (button >= 0 && button < SLANG_COUNT_OF(m_mouseDown)) ? m_mouseDown[button] : false;
    }

    // Returns the current mouse X position.
    float getMouseX() const { return m_mousePos[0]; }
    // Returns the current mouse Y position.
    float getMouseY() const { return m_mousePos[1]; }

    // Window that supplied the mirrored cursor coordinates (or this window
    // before the first cursor event).
    GLFWwindow* getMouseSourceWindow() const;

    // Device management

    // Create a device and retain the missing features if the example requires
    // features that this backend does not support.
    Result createDevice(
        DeviceType deviceType,
        std::vector<Feature> requiredFeatures,
        std::vector<std::pair<std::string, std::string>> preprocessorMacros,
        IDevice** outDevice
    );

    // Window management

    // Creates a window with the specified title and size.
    // Automatically appends device and adapter information to the title.
    Result createWindow(IDevice* device, const char* title, uint32_t width = 640, uint32_t height = 360);
    // Destroys the window.
    void destroyWindow();
    // Returns whether the window has been asked to close.
    bool shouldClose() const;

    // Creates a surface for the window with the specified format.
    // Use Format::Undefined to use the preferred format.
    Result createSurface(IDevice* device, Format format, ISurface** outSurface, bool enableOverlay = true);

public:
#if SLANG_WASM
    bool m_resizePending = false;
#endif
    const char* m_shaderSearchPath = nullptr;
    DeviceType m_deviceType = DeviceType::Default;
    std::vector<Feature> m_missingFeatures;
    GLFWwindow* m_window = nullptr;

    float m_mousePos[2] = {0.0f, 0.0f};
    bool m_mouseDown[3] = {false, false, false};

private:
    bool m_uiVisible = true;
    std::unique_ptr<ExampleOverlay> m_overlay;
#if SLANG_WASM
    ComPtr<IDevice> m_browserDevice;
    ComPtr<ITexture> m_browserImage;
    std::unique_ptr<Blitter> m_browserBlitter;
#endif
};

// Optional presentation of usage and example-owned status text. No window dependency.
class ExampleOverlay
{
public:
    Result init(IDevice* device, Format format, IShaderProgram* program)
    {
        return m_text.init(device, format, program);
    }

    Result render(
        ICommandEncoder* encoder,
        ITexture* image,
        const ExampleUsage& usage,
        std::string_view status,
        float pixelRatio = 1.0f,
        float scale = 1.5f
    )
    {
        if (!std::isfinite(scale) || scale <= 0)
            return SLANG_E_INVALID_ARG;
        float margin = 12 * pixelRatio;
        float fontScale = 1.5f * scale * pixelRatio;
        const auto& size = image->getDesc().size;
        size_t maxColumns =
            size_t(std::max(1.0f, std::floor((float(size.width) - 4 * margin - 2 * fontScale) / (6 * fontScale))));
        std::string content = usage.text();
        if (!status.empty())
            content += std::string("\n\n") + std::string(status);
        // Wrap at word boundaries rather than shrinking the requested text size.
        std::string text;
        std::string_view remaining(content);
        while (!remaining.empty())
        {
            size_t newline = remaining.find('\n');
            auto line = remaining.substr(0, newline);
            while (line.size() > maxColumns)
            {
                size_t split = line.rfind(' ', maxColumns);
                if (split == std::string_view::npos || split == 0)
                    split = maxColumns;
                text.append(line.substr(0, split));
                text += '\n';
                line.remove_prefix(split);
                if (!line.empty() && line.front() == ' ')
                    line.remove_prefix(1);
            }
            text.append(line);
            if (newline == std::string_view::npos)
                break;
            text += '\n';
            remaining.remove_prefix(newline + 1);
        }
        size_t columns = 0, current = 0, lines = 1;
        for (char c : text)
        {
            if (c == '\n')
            {
                columns = std::max(columns, current);
                current = 0;
                ++lines;
            }
            else
                ++current;
        }
        columns = std::max(columns, current);
        m_text.clear();
        SLANG_RETURN_ON_FAIL(m_text.addRect(
            {margin, margin},
            {float(columns * 6 + 2) * fontScale + 2 * margin, float(lines * 9 + 2) * fontScale + 2 * margin},
            {0.003f, 0.006f, 0.012f, 0.85f}
        ));
        SLANG_RETURN_ON_FAIL(m_text.addText(text, {2 * margin, 2 * margin}, fontScale, {0.82f, 0.88f, 0.96f, 1}));
        return m_text.render(encoder, image);
    }

private:
    TextRenderer m_text;
};


namespace detail {
inline void glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos);
inline void glfwWindowSizeCallback(GLFWwindow* window, int width, int height);
inline void glfwWindowIconifyCallback(GLFWwindow* window, int iconified);
inline void glfwWindowMaximizeCallback(GLFWwindow* window, int maximized);
inline void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height);
inline void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos);
inline void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
inline void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
inline void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

inline std::vector<ExampleBase*>& getExamples()
{
    static std::vector<ExampleBase*> examples;
    return examples;
}

inline ExampleBase* mainExample = nullptr;
inline GLFWwindow* mouseSourceWindow = nullptr;

inline void setMainExample(ExampleBase* example)
{
    if (mainExample)
        glfwSetWindowAttrib(mainExample->m_window, GLFW_RESIZABLE, GLFW_FALSE);
    mainExample = example;
    if (mainExample)
        glfwSetWindowAttrib(mainExample->m_window, GLFW_RESIZABLE, GLFW_TRUE);
}

} // namespace detail


inline ExampleBase::ExampleBase() = default;

inline ExampleBase::~ExampleBase()
{
    destroyWindow();
}

inline Result ExampleBase::createDevice(
    DeviceType deviceType,
    std::vector<Feature> requiredFeatures,
    std::vector<std::pair<std::string, std::string>> preprocessorMacros,
    IDevice** outDevice
)
{
    SLANG_RETURN_ON_FAIL(
        rhi::createDevice(
            deviceType,
            requiredFeatures,
            preprocessorMacros,
            outDevice,
            m_shaderSearchPath,
            &m_missingFeatures
        )
    );
    m_deviceType = (*outDevice)->getDeviceType();
    return SLANG_OK;
}

inline Result ExampleBase::createWindow(IDevice* device, const char* title, uint32_t width, uint32_t height)
{
    const auto& deviceInfo = device->getInfo();
    char fullTitle[1024];
    snprintf(
        fullTitle,
        sizeof(fullTitle),
        "%s | %s (%s)",
        title,
        getRHI()->getDeviceTypeName(deviceInfo.deviceType),
        deviceInfo.adapterName
    );

#if SLANG_WASM
    m_browserDevice = device;
    m_browserBlitter = std::make_unique<Blitter>(device);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
#else
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
#endif

    m_window = glfwCreateWindow(width, height, fullTitle, nullptr, nullptr);
    if (!m_window)
    {
        return SLANG_FAIL;
    }
    glfwSetWindowUserPointer(m_window, this);
    glfwSetWindowPosCallback(m_window, detail::glfwWindowPosCallback);
    glfwSetWindowSizeCallback(m_window, detail::glfwWindowSizeCallback);
    glfwSetWindowIconifyCallback(m_window, detail::glfwWindowIconifyCallback);
    glfwSetWindowMaximizeCallback(m_window, detail::glfwWindowMaximizeCallback);
    glfwSetFramebufferSizeCallback(m_window, detail::glfwFramebufferSizeCallback);
    glfwSetCursorPosCallback(m_window, detail::glfwCursorPosCallback);
    glfwSetMouseButtonCallback(m_window, detail::glfwMouseButtonCallback);
    glfwSetScrollCallback(m_window, detail::glfwScrollCallback);
    glfwSetKeyCallback(m_window, detail::glfwKeyCallback);

    return SLANG_OK;
}

inline GLFWwindow* ExampleBase::getMouseSourceWindow() const
{
    return detail::mouseSourceWindow ? detail::mouseSourceWindow : m_window;
}

inline void ExampleBase::destroyWindow()
{
    if (m_window)
    {
        if (detail::mouseSourceWindow == m_window)
            detail::mouseSourceWindow = nullptr;
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
}

inline bool ExampleBase::shouldClose() const
{
    return m_window && glfwWindowShouldClose(m_window) != 0;
}

inline Result ExampleBase::createSurface(IDevice* device, Format format, ISurface** outSurface, bool enableOverlay)
{
    ASSERT(m_window, "Window must be created before creating surface");

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);

#if SLANG_WASM
    SLANG_RETURN_ON_FAIL(device->createSurface(WindowHandle::fromWGPUCanvas("#canvas"), outSurface));
#else
    SLANG_RETURN_ON_FAIL(device->createSurface(getWindowHandleFromGLFW(m_window), outSurface));
#endif
    SurfaceConfig surfaceConfig;
    surfaceConfig.width = width;
    surfaceConfig.height = height;
    surfaceConfig.format = format;
    SLANG_RETURN_ON_FAIL((*outSurface)->configure(surfaceConfig));
    if (enableOverlay && device->hasFeature(Feature::Rasterization))
    {
        ComPtr<IShaderProgram> program;
        SLANG_RETURN_ON_FAIL(
            createProgram(device, "../base/text-renderer.slang", {"textVertex", "textFragment"}, program.writeRef())
        );
        m_overlay = std::make_unique<ExampleOverlay>();
        SLANG_RETURN_ON_FAIL(m_overlay->init(device, (*outSurface)->getConfig()->format, program));
    }


    return SLANG_OK;
}

inline Result ExampleBase::acquireFrame(ISurface* surface, ITexture** image)
{
#if SLANG_WASM
    *image = nullptr;
    const auto* config = surface->getConfig();
    if (!config)
        return SLANG_OK;
    if (!m_browserImage || m_browserImage->getDesc().size.width != config->width ||
        m_browserImage->getDesc().size.height != config->height || m_browserImage->getDesc().format != config->format)
    {
        TextureDesc desc = {};
        desc.size = {config->width, config->height, 1};
        desc.format = config->format;
        desc.usage = TextureUsage::RenderTarget | TextureUsage::ShaderResource;
        SLANG_RETURN_ON_FAIL(m_browserDevice->createTexture(desc, nullptr, m_browserImage.writeRef()));
    }
    *image = m_browserImage.get();
    (*image)->addRef();
    return SLANG_OK;
#else
    return surface->acquireNextImage(image);
#endif
}

inline Result ExampleBase::presentFrame(ISurface* surface, ICommandQueue* queue)
{
#if SLANG_WASM
    // This final pass has no uniform uploads or readbacks, so it never suspends
    // with a canvas texture acquired. All scene work was submitted offscreen.
    auto encoder = queue->createCommandEncoder();
    ComPtr<ITexture> image;
    SLANG_RETURN_ON_FAIL(surface->acquireNextImage(image.writeRef()));
    if (!image)
        return SLANG_OK;
    SLANG_RETURN_ON_FAIL(m_browserBlitter->blit(image, m_browserImage, encoder));
    ComPtr<ICommandBuffer> commands;
    SLANG_RETURN_ON_FAIL(encoder->finish(commands.writeRef()));
    SLANG_RETURN_ON_FAIL(queue->submit(commands));
#endif
    return surface->present();
}

inline bool ExampleBase::isMouseHovered() const
{
#if SLANG_WASM
    return EM_ASM_INT({ return Module['canvas'].matches(':hover'); });
#else
    return glfwGetWindowAttrib(getMouseSourceWindow(), GLFW_HOVERED) != 0;
#endif
}

inline Result ExampleBase::drawUsage(ICommandEncoder* encoder, ITexture* image, std::string_view status, float scale)
{
    if (!m_uiVisible || !m_overlay)
        return SLANG_OK;
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    float pixelRatio = float(image->getDesc().size.width) / std::max(width, 1);
    return m_overlay->render(encoder, image, getUsage(), status, pixelRatio, scale);
}

namespace detail {

inline bool layoutInProgress = false;

inline void layoutWindows()
{
    if (getExamples().size() <= 1)
    {
        return;
    }

#if SLANG_WASM
    return;
#else
    static const int kMargin = 100;

    int wx, wy, ww, wh;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    glfwGetMonitorWorkarea(monitor, &wx, &wy, &ww, &wh);

    int frameLeft, frameTop, frameRight, frameBottom;
    glfwGetWindowFrameSize(mainExample->m_window, &frameLeft, &frameTop, &frameRight, &frameBottom);

    layoutInProgress = true;

    int x = wx + kMargin;
    int y = wy + kMargin;

    for (ExampleBase* example : getExamples())
    {
        int width, height;
        glfwGetWindowSize(example->m_window, &width, &height);
        width += frameLeft + frameRight;
        height += frameTop + frameBottom;

        if (x + width >= ww)
        {
            x = wx + kMargin;
            y += height;
        }
        glfwSetWindowPos(example->m_window, x, y);
        x += width;
    }

    layoutInProgress = false;
#endif
}

inline void glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos)
{
    if (layoutInProgress)
    {
        return;
    }
    ExampleBase* srcExample = (ExampleBase*)glfwGetWindowUserPointer(window);
    if (srcExample == mainExample)
    {
        layoutWindows();
    }
}

inline void glfwWindowSizeCallback(GLFWwindow* window, int width, int height)
{
    if (layoutInProgress)
    {
        return;
    }
    ExampleBase* srcExample = (ExampleBase*)glfwGetWindowUserPointer(window);
    for (ExampleBase* example : getExamples())
    {
        if (example != srcExample)
        {
            glfwSetWindowSize(example->m_window, width, height);
        }
    }
    if (srcExample == mainExample)
    {
        layoutWindows();
    }
}

inline void glfwWindowIconifyCallback(GLFWwindow* window, int iconified)
{
    ExampleBase* srcExample = (ExampleBase*)glfwGetWindowUserPointer(window);
    for (ExampleBase* example : getExamples())
    {
        if (example != srcExample)
        {
            if (iconified)
            {
                glfwIconifyWindow(example->m_window);
            }
            else
            {
                glfwRestoreWindow(example->m_window);
            }
        }
    }
}

inline void glfwWindowMaximizeCallback(GLFWwindow* window, int maximized)
{
    ExampleBase* srcExample = (ExampleBase*)glfwGetWindowUserPointer(window);
    for (ExampleBase* example : getExamples())
    {
        if (example != srcExample)
        {
            if (maximized)
            {
                glfwMaximizeWindow(example->m_window);
            }
            else
            {
                glfwRestoreWindow(example->m_window);
            }
        }
    }
}

inline void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
#if SLANG_WASM
    // Browser callbacks must not initiate a second Asyncify stack while the main
    // loop is suspended. Apply resizes from that loop before acquiring a frame.
    static_cast<ExampleBase*>(glfwGetWindowUserPointer(window))->m_resizePending = true;
#else
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    ExampleBase* example = (ExampleBase*)glfwGetWindowUserPointer(window);
    example->onResize(windowWidth, windowHeight, framebufferWidth, framebufferHeight);
#endif
}

inline void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    mouseSourceWindow = window;
    for (ExampleBase* example : getExamples())
    {
        example->m_mousePos[0] = xpos;
        example->m_mousePos[1] = ypos;
        example->onMousePosition(xpos, ypos);
    }
}

inline void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    for (ExampleBase* example : getExamples())
    {
        if (button < SLANG_COUNT_OF(example->m_mouseDown))
            example->m_mouseDown[button] = (action == GLFW_PRESS);
        example->onMouseButton(button, action, mods);
    }
}

inline void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    for (ExampleBase* example : getExamples())
    {
        example->onScroll(xoffset, yoffset);
    }
}

inline void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    for (ExampleBase* example : getExamples())
    {
        if (key == GLFW_KEY_F1)
        {
            if (action == GLFW_PRESS)
                example->toggleUI();
            continue;
        }
        example->onKey(key, scancode, action, mods);
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(example->m_window, GLFW_TRUE);
        }
    }
}

inline int runExamples(int argc, const char** argv, ExampleBase* (*createExample)(), const char* shaderSearchPath)
{
    std::vector<DeviceType> deviceTypes = {
#if SLANG_WASM
        DeviceType::WGPU,
#else
        DeviceType::D3D11,
        DeviceType::D3D12,
        DeviceType::Vulkan,
        DeviceType::Metal,
        DeviceType::CPU,
        DeviceType::CUDA,
        DeviceType::WGPU,
#endif
    };

    if (argc > 1)
    {
        if (argc != 3 || std::string(argv[1]) != "--device")
        {
            LOG_ERROR("Usage: %s [--device <d3d11|d3d12|vulkan|metal|cpu|cuda|wgpu>]", argv[0]);
            return 1;
        }
        std::string requestedDevice = argv[2];
        auto toLower = [](unsigned char c)
        {
            return char(std::tolower(c));
        };
        std::transform(requestedDevice.begin(), requestedDevice.end(), requestedDevice.begin(), toLower);
        auto selectedDevice = std::find_if(
            deviceTypes.begin(),
            deviceTypes.end(),
            [&](DeviceType type)
            {
                std::string name = getRHI()->getDeviceTypeName(type);
                std::transform(name.begin(), name.end(), name.begin(), toLower);
                return name == requestedDevice;
            }
        );
        if (selectedDevice == deviceTypes.end())
        {
            LOG_ERROR("Unknown device: %s", argv[2]);
            return 1;
        }
        deviceTypes = {*selectedDevice};
    }

    if (!glfwInit())
    {
        LOG_ERROR("Failed to initialize GLFW");
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    std::vector<ExampleBase*>& examples = getExamples();
    int exitCode = 0;

    bool usagePrinted = false;

    // Create an example for each supported device type
    for (DeviceType deviceType : deviceTypes)
    {
        if (rhi::getRHI()->isDeviceTypeSupported(deviceType))
        {
            ExampleBase* example = createExample();
            example->m_shaderSearchPath = shaderSearchPath;
            if (!usagePrinted)
            {
                LOG_INFO("%s", example->getUsage().text().c_str());
                usagePrinted = true;
            }
            Result result = example->init(deviceType);
            if (SLANG_FAILED(result))
            {
                if (result == SLANG_E_NOT_AVAILABLE && !example->m_missingFeatures.empty())
                {
                    std::string missingFeatures;
                    for (Feature feature : example->m_missingFeatures)
                    {
                        if (!missingFeatures.empty())
                            missingFeatures += ", ";
                        const char* name = getRHI()->getFeatureName(feature);
                        missingFeatures += name ? name : "unknown";
                    }
                    LOG_INFO(
                        "%s: skipped (missing required features: %s).",
                        getRHI()->getDeviceTypeName(deviceType),
                        missingFeatures.c_str()
                    );
                }
                else
                {
                    exitCode = 1;
                    LOG_ERROR(
                        "Could not initialize %s example (0x%08x).",
                        getRHI()->getDeviceTypeName(deviceType),
                        unsigned(result)
                    );
                }
                delete example;
                continue;
            }
            examples.push_back(example);
            if (!mainExample)
                setMainExample(example);
        }
        else
        {
            LOG_INFO("%s: skipped (backend not enabled in this build).", getRHI()->getDeviceTypeName(deviceType));
        }
    }

    layoutWindows();

    if (examples.empty())
        exitCode = 1;

#if SLANG_WASM
    EM_ASM({
        Module['rhiReady'] = true;
        Module['rhiFrameCount'] = 0;
    });
#endif
    while (!examples.empty())
    {
        glfwPollEvents();

        double time = glfwGetTime();
        bool shouldClose = false;

        for (auto it = examples.begin(); it != examples.end();)
        {
            ExampleBase* example = *it;
            shouldClose |= example->shouldClose();
#if SLANG_WASM
            double cssWidth, cssHeight;
            // GLFW sets the canvas CSS size when high-DPI support is enabled.
            // Measure its container so those inline styles do not prevent resizing.
            emscripten_get_element_css_size("#viewport", &cssWidth, &cssHeight);
            int width, height;
            glfwGetWindowSize(example->m_window, &width, &height);
            if (width != int(cssWidth) || height != int(cssHeight))
                glfwSetWindowSize(example->m_window, int(cssWidth), int(cssHeight));
            if (example->m_resizePending)
            {
                int framebufferWidth, framebufferHeight;
                glfwGetWindowSize(example->m_window, &width, &height);
                glfwGetFramebufferSize(example->m_window, &framebufferWidth, &framebufferHeight);
                example->m_resizePending = false;
                example->onResize(width, height, framebufferWidth, framebufferHeight);
            }
#endif
            Result result = example->update(time);
            if (SLANG_SUCCEEDED(result))
                result = example->draw();
            if (SLANG_FAILED(result))
            {
                LOG_ERROR(
                    "%s: example frame failed (0x%08x).",
                    getRHI()->getDeviceTypeName(example->m_deviceType),
                    unsigned(result)
                );
                exitCode = 1;
                it = examples.erase(it);
                if (mainExample == example)
                    setMainExample(examples.empty() ? nullptr : examples.front());
                example->shutdown();
                delete example;
                layoutWindows();
            }
            else
                ++it;
        }
        if (shouldClose)
            break;
#if SLANG_WASM
        EM_ASM({ ++Module['rhiFrameCount']; });
        // Keep the native lifetime while yielding to browser events between frames.
        emscripten_sleep(16);
#endif
    }

    for (ExampleBase* example : examples)
    {
        example->shutdown();
        delete example;
    }

    examples.clear();
    mainExample = nullptr;

    glfwTerminate();
#if SLANG_WASM
    EM_ASM({ Module['rhiStopped'] = true; });
#endif
    return exitCode;
}

} // namespace detail

} // namespace rhi

#define EXAMPLE_MAIN(Example)                                                                                          \
    int main(int argc, const char** argv)                                                                              \
    {                                                                                                                  \
        return rhi::detail::runExamples(                                                                               \
            argc,                                                                                                      \
            argv,                                                                                                      \
            []() -> rhi::ExampleBase*                                                                                  \
            {                                                                                                          \
                return new Example();                                                                                  \
            },                                                                                                         \
            EXAMPLE_DIR                                                                                                \
        );                                                                                                             \
    }
