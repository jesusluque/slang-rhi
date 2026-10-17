// This file implements a Metal kernel that applies an indirect draw count.

#include <metal_stdlib>
using namespace metal;

struct DrawCountParams
{
    uint wordsPerDraw;
    uint maxDrawCount;
};

// Copies indirect draw arguments and zeroes the draws at or above the draw count,
// so that issuing maxDrawCount indirect draws only renders the first `count` draws.
kernel void apply_draw_count(
    device const uint* args [[buffer(0)]],
    device uint* outArgs [[buffer(1)]],
    device const uint* count [[buffer(2)]],
    constant DrawCountParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]
)
{
    if (index >= params.maxDrawCount)
        return;
    bool enabled = index < count[0];
    for (uint i = 0; i < params.wordsPerDraw; ++i)
    {
        uint word = index * params.wordsPerDraw + i;
        outArgs[word] = enabled ? args[word] : 0;
    }
}
