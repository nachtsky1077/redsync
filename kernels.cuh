#define CUB_STDERR

#include <cuda.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

// custom reduce op for computing mean(abs(tensor))
struct AbsSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return abs(a)+abs(b);
    }
};

// custom reduce op for computing max(abs(tensor))
struct AbsMax
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return abs(b) > abs(a) ? abs(b) : abs(a);
    }
};

// Functor type for selecting values less than some threshold
struct LargerThan
{
    float compare;
    __host__ __device__ __forceinline__
    LargerThan(float compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const float &a) const {
        return (a > compare);
    }
};

void MeanKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items) {
    AbsSum sum_op;
    float init = 0.0f;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, sum_op, init));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, sum_op, init));

}

void MaxKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items) {
    AbsMax max_op;
    float init = FLT_MIN;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
}

void CountNonZero(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int *num_out, float threshold, int num_items) {
    LargerThan select_op(threshold);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op));
}

