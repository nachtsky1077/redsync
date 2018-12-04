include "kernels.h"

void MeanKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items) {
    AbsSum sum_op;
    float init = 0.0f;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, sum_op, init));
    cub::CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    cub::CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, sum_op, init));

}

void MaxKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items) {
    AbsMax max_op;
    float init = FLT_MIN;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
    cub::CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    cub::CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
}

void CountNonZero(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int *num_out, float threshold, int num_items) {
    LargerThan select_op(threshold);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op, init));
    cub::CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    cub::CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op, init));
}