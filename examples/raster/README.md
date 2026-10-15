# Raster example

This example renders the Slang logo and a grid of material spheres using indexed
geometry, instancing, shadow mapping, metallic/roughness shading, and procedural
environment lighting. It records an HDR scene with optional 4x MSAA, compute bloom,
and tone mapping. Shader cursors bind resources and debug groups label GPU passes.

## Build and run

From the repository root, with examples enabled:

```sh
cmake --preset default
cmake --build build --config Debug --target example-raster
./build/Debug/example-raster --device vulkan
```

On Windows, use a developer shell for MSVC and append `.exe` to the executable.
The example requires surface, rasterization, and compute support. Select a backend
with `--device d3d11`, `d3d12`, `vulkan`, `metal`, or `wgpu`, subject to the platform
and build configuration. Omit `--device` to open all compatible backend windows.
Input is mirrored across windows and frames are synchronized by default;
`--free-running` lets each backend advance independently.

## Controls

| Input | Action |
| --- | --- |
| Left drag / right drag | Orbit camera / move light |
| Mouse wheel | Zoom |
| `[` / `]` | Decrease / increase roughness |
| `-` / `=` | Decrease / increase exposure |
| `B` | Toggle bloom |
| `E` | Switch studio / outdoor environment |
| `,` / `.` | Rotate environment |
| `I` | Toggle environment lighting |
| `M` | Toggle 4x MSAA when supported |
| `S` | Toggle material spheres |
| `R` | Reset settings |
| `F1` / `F2` | Toggle overlay / fit windows |
| `Esc` | Close |

The overlay reports application-loop FPS and frame time.

## Sources

`example-raster.cpp` creates resources and records rendering passes.
`raster.slang` shades the scene and shadows; `environment.slang` generates and
filters environment maps using `environment-common.slang`; `post.slang` implements
bloom and tone mapping. Camera, sphere geometry, and frame statistics
helpers live in `../base`.

The bundled logo mesh, source artwork, and regeneration instructions are documented
in [the asset README](../base/assets/README.md). Environment maps are generated at
startup; no external scene assets are needed at runtime.
