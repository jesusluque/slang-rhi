#pragma once

#include "math.h"
#include <algorithm>
#include <array>
#include <cstdint>

namespace rhi {

using namespace math;

struct Camera
{
    uint32_t width{100};
    uint32_t height{100};
    float aspectRatio{1.0f};
    float3 position{1, 1, 1};
    float3 target{0, 0, 0};
    float3 fwd;
    float3 right;
    float3 up{0, 1, 0};
    float fov{70.0f};

    float3 imageU;
    float3 imageV;
    float3 imageW;

    Camera() { recompute(); }

    void recompute()
    {
        aspectRatio = float(width) / float(height);

        fwd = normalize(target - position);
        right = normalize(cross(fwd, up));
        up = normalize(cross(right, fwd));

        float fovRad = fov * (3.14159265358979323846f / 180.0f);

        imageU = right * std::tan(fovRad * 0.5f) * aspectRatio;
        imageV = up * std::tan(fovRad * 0.5f);
        imageW = fwd;
    }
};


inline float3 orbitDirection(float yaw, float pitch)
{
    return {std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
}

struct FlyCameraController
{
    Camera* camera = nullptr;
    bool mouseDown{false};
    float2 mousePos;
    enum class Movement
    {
        Left,
        Right,
        Up,
        Down,
        Forward,
        Backward
    };
    std::array<bool, 6> movement = {};
    bool shiftDown{false};
    float3 moveDelta{0.f};
    float2 rotateDelta{0.f};
    float moveSpeed{1.0f};
    float rotateSpeed{0.002f};

    static constexpr float kMoveShiftFactor = 10.0f;
    void setCamera(Camera* value) { camera = value; }
    void setMovement(Movement direction, bool down)
    {
        movement[size_t(direction)] = down;
        moveDelta = {
            float(movement[1]) - float(movement[0]),
            float(movement[2]) - float(movement[3]),
            float(movement[4]) - float(movement[5])
        };
    }
    void setFast(bool down) { shiftDown = down; }
    void setRotating(bool down) { mouseDown = down; }

    bool update(float dt)
    {
        if (!camera)
            return false;
        bool changed = false;
        float3 position = camera->position;
        float3 fwd = camera->fwd;
        float3 up = camera->up;
        float3 right = camera->right;

        // Move
        if (length(moveDelta) > 0)
        {
            float3 offset = right * moveDelta.x;
            offset += up * moveDelta.y;
            offset += fwd * moveDelta.z;
            float factor = shiftDown ? kMoveShiftFactor : 1.0f;
            offset = normalize(offset) * moveSpeed * factor * dt;
            position += offset;
            changed = true;
        }

        // Rotate
        if (length(rotateDelta) > 0)
        {
            float yaw = std::atan2(fwd.z, fwd.x);
            float pitch = std::asin(std::clamp(fwd.y, -1.0f, 1.0f));
            yaw += rotateSpeed * rotateDelta.x;
            const float maxPitch = 3.14159265358979323846f * 0.5f - 1e-3f;
            pitch = std::clamp(pitch - rotateSpeed * rotateDelta.y, -maxPitch, maxPitch);
            fwd = float3(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
            rotateDelta = float2();
            changed = true;
        }

        if (changed)
        {
            camera->position = position;
            camera->target = position + fwd;
            camera->up = float3(0, 1, 0);
            camera->recompute();
        }

        return changed;
    }

    void onMousePosition(float x, float y)
    {
        float2 pos = float2(x, y);
        if (mouseDown)
        {
            rotateDelta += pos - mousePos;
        }
        mousePos = pos;
    }
};

// Orbit state belongs to the caller, allowing camera presets and reset controls.
struct OrbitCameraController
{
    float sensitivity = 0.006f;
    float minPitch = 0.03f, maxPitch = 1.3f;
    float minDistance = 4.0f, maxDistance = 20.0f;

    void rotate(float2 delta, float& yaw, float& pitch) const
    {
        yaw -= delta.x * sensitivity;
        pitch = std::clamp(pitch + delta.y * sensitivity, minPitch, maxPitch);
    }
    void zoom(float scroll, float& distance) const
    {
        distance = std::clamp(distance * std::exp(-0.1f * scroll), minDistance, maxDistance);
    }
};

} // namespace rhi
