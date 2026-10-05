// The kernel `MetalOrtDetector` reads a YOLOv8 or YOLO11 output with, the
// Metal counterpart of `best_class` in `platform::cuda::driver::ptx::FIT_PTX`:
// of `pictures` outputs of `rows` (four, then a score per class) by `boxes`
// floats, in buffer 1, it writes each box to buffer 2 as six floats — its
// centre, width and height, and its best class's score and number. A strict
// `>` keeps the first of equal scores, as the CPU's reading does. A thread a
// box, so neighbouring threads read neighbouring floats of each row.

#include <metal_stdlib>
using namespace metal;

struct Size {
    uint rows;
    uint boxes;
    uint pictures;
    uint unused;
};

kernel void best_class(constant Size &size [[buffer(0)]],
                       device const float *output [[buffer(1)]],
                       device float *best [[buffer(2)]],
                       uint2 id [[thread_position_in_grid]]) {
    uint box = id.x;
    uint picture = id.y;
    if (box >= size.boxes || picture >= size.pictures) {
        return;
    }
    device const float *rows = output + picture * size.rows * size.boxes + box;
    float score = rows[4 * size.boxes];
    uint class_id = 0;
    for (uint row = 5; row < size.rows; row++) {
        float candidate = rows[row * size.boxes];
        if (candidate > score) {
            score = candidate;
            class_id = row - 4;
        }
    }
    device float *written = best + (picture * size.boxes + box) * 6;
    written[0] = rows[0];
    written[1] = rows[size.boxes];
    written[2] = rows[2 * size.boxes];
    written[3] = rows[3 * size.boxes];
    written[4] = score;
    written[5] = float(class_id);
}
