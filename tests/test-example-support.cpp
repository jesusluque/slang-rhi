#include "testing.h"
#include "../examples/base/camera.h"
#include "../examples/base/utils.h"
#include "../examples/base/geometry.h"

using namespace rhi;

TEST_CASE("example-camera")
{
    Camera camera;
    camera.position = {0, 0, 5};
    camera.target = {0, 0, 0};
    camera.width = 200;
    camera.height = 100;
    camera.fov = 90;
    camera.up = {0, 1, 0};
    camera.recompute();
    CHECK(camera.imageU.x == doctest::Approx(2));
    CHECK(camera.imageV.y == doctest::Approx(1));
    CHECK(camera.imageW.z == doctest::Approx(-1));
    auto view = math::lookAt(camera.position, camera.target);
    auto eye = math::mul(view, math::float4(camera.position, 1));
    CHECK(eye.x == doctest::Approx(0));
    CHECK(eye.y == doctest::Approx(0));
    CHECK(eye.z == doctest::Approx(0));
    auto projection = math::perspective(2, 1.0f, 0.25f, 100);
    auto nearClip = math::mul(projection, math::float4(0, 0, -0.25f, 1));
    auto farClip = math::mul(projection, math::float4(0, 0, -100, 1));
    CHECK(nearClip.z / nearClip.w == doctest::Approx(0));
    CHECK(farClip.z / farClip.w == doctest::Approx(1));
}

TEST_CASE("example-primitive-geometry")
{
    auto quad = geometry::Mesh::createQuad({2, 4});
    CHECK(quad.vertexCount() == 4);
    CHECK(quad.indexCount() == 6);
    auto cube = geometry::Mesh::createCube({2, 4, 6});
    CHECK(cube.vertexCount() == 24);
    CHECK(cube.indexCount() == 36);
    for (const auto& mesh : {quad, cube})
    {
        for (size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            auto a = mesh.vertices[mesh.indices[i]];
            auto b = mesh.vertices[mesh.indices[i + 1]];
            auto c = mesh.vertices[mesh.indices[i + 2]];
            CHECK(math::dot(math::cross(b.position - a.position, c.position - a.position), a.normal) > 0);
        }
    }
    for (const auto& vertex : cube.vertices)
    {
        CHECK(std::abs(vertex.position.x) == 1);
        CHECK(std::abs(vertex.position.y) == 2);
        CHECK(std::abs(vertex.position.z) == 3);
    }
    auto sphere = geometry::Mesh::createSphere(8, 4);
    CHECK(sphere.vertices.size() == 43);
    CHECK(sphere.indices.size() == 144);
    for (const auto& vertex : sphere.vertices)
        CHECK(math::length(vertex.normal) == doctest::Approx(1));
    CHECK(geometry::Mesh::createSphere(2, 4).vertices.empty());
}

TEST_CASE("example-fixed-step-clock")
{
    FixedStepClock clock(0.1, 3);
    CHECK(clock.update(0, true) == 0);
    CHECK(clock.update(0.25, true) == 2);
    CHECK(clock.update(0.3, true) == 1);
    CHECK(clock.update(100, true) == 3);
    CHECK(clock.update(101, false) == 0);
    CHECK(clock.update(101.05, true) == 0);
    clock.reset();
    CHECK(clock.update(101.1, true) == 0);
    CHECK(clock.update(101.15, true) == 1);
}

TEST_CASE("example-usage")
{
    ExampleUsage usage = {"Example description", "Space: Pause"};
    auto text = usage.text();
    CHECK(text.find(usage.description) != std::string::npos);
    CHECK(text.find(usage.controls) != std::string::npos);
    CHECK(text.find("F1: Show/hide UI") != std::string::npos);
    CHECK(text.find("Esc: Close") != std::string::npos);
}

TEST_CASE("example-camera-controllers")
{
    Camera camera;
    camera.position = {0, 0, 5};
    camera.target = {0, 0, 0};
    camera.up = {0, 1, 0};
    camera.recompute();
    FlyCameraController fly;
    fly.setCamera(&camera);
    fly.setMovement(FlyCameraController::Movement::Forward, true);
    REQUIRE(fly.update(1));
    CHECK(camera.position.z == doctest::Approx(4));
    fly.setMovement(FlyCameraController::Movement::Forward, false);
    CHECK_FALSE(fly.update(1));
    fly.onMousePosition(100, 100);
    fly.setRotating(true);
    fly.onMousePosition(110, 100);
    REQUIRE(fly.update(0));
    CHECK(camera.fwd.x > 0);
    CHECK_FALSE(fly.update(0));
    OrbitCameraController orbit;
    float yaw = 0, pitch = 0.5f, distance = 10;
    orbit.rotate({10, 10000}, yaw, pitch);
    CHECK(yaw == doctest::Approx(-0.06f));
    CHECK(pitch == orbit.maxPitch);
    orbit.zoom(100, distance);
    CHECK(distance == orbit.minDistance);
}

TEST_CASE("example-sphere-uvs")
{
    for (uint32_t slices : {3u, 8u})
        for (uint32_t stacks : {2u, 4u})
        {
            auto mesh = geometry::Mesh::createSphere(slices, stacks);
            REQUIRE(mesh.indices.size() == 6 * slices * (stacks - 1));
            for (const auto& vertex : mesh.vertices)
            {
                CHECK(vertex.uv.x >= 0);
                CHECK(vertex.uv.x <= 1);
                CHECK(vertex.uv.y >= 0);
                CHECK(vertex.uv.y <= 1);
                CHECK(vertex.position.y == doctest::Approx(std::cos(vertex.uv.y * 3.14159265f)).epsilon(1e-5));
                CHECK(math::length(vertex.normal) == doctest::Approx(1));
            }
            for (uint32_t y = 1; y < stacks; ++y)
            {
                const auto& first = mesh.vertices[(y - 1) * (slices + 1)];
                const auto& last = mesh.vertices[y * (slices + 1) - 1];
                CHECK(math::length(first.position - last.position) == 0);
                CHECK(math::length(first.normal - last.normal) == 0);
                CHECK(first.uv.x == 0);
                CHECK(last.uv.x == 1);
                CHECK(first.uv.y == last.uv.y);
            }
            for (size_t i = 0; i < mesh.indices.size(); i += 3)
            {
                for (size_t j = 0; j < 3; ++j)
                    REQUIRE(mesh.indices[i + j] < mesh.vertices.size());
                const auto& a = mesh.vertices[mesh.indices[i]];
                const auto& b = mesh.vertices[mesh.indices[i + 1]];
                const auto& c = mesh.vertices[mesh.indices[i + 2]];
                CHECK(math::dot(math::cross(b.position - a.position, c.position - a.position), a.position) > 0);
                auto ab = b.uv - a.uv;
                auto ac = c.uv - a.uv;
                // No collapsed UV triangles or interpolation across the wrap seam.
                CHECK(ab.x * ac.y - ab.y * ac.x > 0);
                float minU = std::min({a.uv.x, b.uv.x, c.uv.x});
                float maxU = std::max({a.uv.x, b.uv.x, c.uv.x});
                CHECK(maxU - minU <= 1.0f / slices + 1e-6f);
            }
        }
    CHECK(geometry::Mesh::createSphere(8, 1).vertices.empty());
}
