# Examples

These examples demonstrate slang-rhi resource creation, shader binding, and GPU
command recording across graphics backends. Start with `surface` or `triangle`
for the basics, then explore the larger rendering examples.

| Example | Demonstrates |
| --- | --- |
| [surface](surface/example-surface.cpp) | Creating a window and surface, clearing an image, and presenting it. |
| [triangle](triangle/example-triangle.cpp) | Vertex buffers, a graphics pipeline, and a triangle draw. |
| [shader-toy](shader-toy/example-shader-toy.cpp) | Rendering procedural images with compute shaders; use the arrow keys to switch shaders. |
| [raster](raster/example-raster.cpp) | A Slang logo and instanced spheres with shadow mapping, environment lighting, MSAA, bloom, and tone mapping. |
| [particles](particles/example-particles.cpp) | GPU particle simulation, indirect drawing, mouse interaction, and bloom around the Slang logo. |
| [path-tracer](path-tracer/example-path-tracer.cpp) | Progressive path tracing with acceleration structures, ray tracing pipelines or ray queries, and sample accumulation. |

## Build and run

From the repository root:

```sh
cmake --preset default
cmake --build build --config Debug --target example-raster
./build/Debug/example-raster --device vulkan
```

Replace `example-raster` with any `example-<name>` target listed above. Examples
are enabled by `SLANG_RHI_BUILD_EXAMPLES`. On Windows, executables have an `.exe`
suffix; use a developer shell when building with MSVC.

`--device` selects `d3d11`, `d3d12`, `vulkan`, `metal`, `cpu`, `cuda`, or `wgpu`,
subject to platform, build configuration, and the example's required features.
The raster example requires rasterization and compute support; the path tracer
requires acceleration structures and ray tracing or ray query support.

Omit `--device` to open windows for the supported backends. Input is mirrored
across windows, and frames are synchronized by default. Use `--free-running`
to let each backend advance independently.

Examples with an overlay display their controls on screen. `F1` toggles the
overlay, `F2` fits the windows to the primary example's monitor, and `Esc` closes
the examples.

## Smoke tests

Use `--frames N` to run an example for a fixed number of frames per backend and
exit through its normal shutdown path:

```sh
./build/Debug/example-raster --device vulkan --frames 3
```

This also works with multiple backend windows and `--free-running`. Initialization
or frame errors, stopping before the frame limit, and having no compatible backend
produce a nonzero exit code. Backends missing required features are skipped.

When examples and tests are enabled, CTest runs each desktop example for three
frames on the available compatible backends:

```sh
cmake --build build --config Debug
ctest --test-dir build -C Debug -L examples --output-on-failure
```

These smoke tests open normal windows and require a graphical session. They check
startup, frame execution, and shutdown; they do not compare rendered images.

## Source layout

Each example's `example-<name>.cpp` is its entry point, with Slang shaders next
to it. The [base](base) directory provides shared windowing, camera, geometry,
text, and profiling helpers. The bundled logo's source and regeneration steps
are documented in [base/assets/README.md](base/assets/README.md).

The [wasm](wasm) directory contains a separate WebGPU browser test, built as
`wasm-test` with the `emscripten` preset.
