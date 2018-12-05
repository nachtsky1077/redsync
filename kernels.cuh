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

// Functor type for selecting values less than some threshold
struct LargerThanKV
{
    float compare;
    __host__ __device__ __forceinline__
    LargerThanKV(float compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const float2 &a) const {
        return (a.y > compare);
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
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

}

void MaxKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items) {
    AbsMax max_op;
    float init = 0.0f;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, in, out, num_items, max_op, init));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

void CountNonZero(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int *num_out, float threshold, int num_items) {
    LargerThan select_op(threshold);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, in, out, num_out, num_items, select_op));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

__global__ void GetKV(float2 *in, int *keys, float *vals, int num_items) {
    size_t STRIDE = gridDim.x * blockDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < num_items; i+=STRIDE) {
        keys[i] = in[i].x;
        vals[i] = in[i].y;
    }
}

__global__ void CreateKV(int *keys, float *vals, float2 *out, int num_items) {
    size_t STRIDE = gridDim.x * blockDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < num_items; i+=STRIDE) {
        float2 kv_pair = make_float2(keys[i], vals[i]);
        out[i] = kv_pair;
    }
}

template<int NUM_THREADS>
void GetFiltered(cub::CachingDeviceAllocator &g_allocator, float *in_vals, int *in_keys, float *out_vals, int *out_keys, int *num_out, float threshold, int num_items) {
    float2 *input_kvs = NULL;
    float2 *output_kvs = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&input_kvs, sizeof(float2)*num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&output_kvs, sizeof(float2)*num_items));
    CreateKV<<<8, NUM_THREADS>>>(in_keys, in_vals, input_kvs, num_items);

    LargerThanKV select_op(threshold);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_kvs, output_kvs, num_out, num_items, select_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_kvs, output_kvs, num_out, num_items, select_op));

    GetKV<<<8, NUM_THREADS>>>(output_kvs, out_keys, out_vals, num_items);

    if (input_kvs) CubDebugExit(g_allocator.DeviceFree(input_kvs));
    if (output_kvs) CubDebugExit(g_allocator.DeviceFree(output_kvs));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

// TODO: add multi-stream pipelining
void TrimmedTopK(cub::CachingDeviceAllocator &g_allocator, float *in, float *out_vals, int *out_indices, int *out_num, int k, int num_items) {
    // create all device variables.
    float *d_in_vals = NULL;
    float *d_in_indices = NULL;
    float *d_out_max = NULL;
    float *d_out_sum = NULL;
    int *d_out_num = NULL;
    float *d_out_vals = NULL;
    int *d_out_indices = NULL;

    // allocate and memcpy.
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in_vals, sizeof(float)*num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in_indices, sizeof(int)*num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_max, sizeof(float)*1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_sum, sizeof(float)*1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_num, sizeof(int)*1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_vals, sizeof(float)*num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_indices, sizeof(int)*num_items));

    CubDebugExit(cudaMemcpy(d_in, in, num_items*sizeof(float), cudaMemcpyHostToDevice));
    // compute mean and max
    // epsilon = 0.2
    // ratio = 1-epsilon
    // count non-zero abs(X) > threshold
    // while nnz > k
    //  threshold = mean + ratio * (max - mean)
    //  nnz = count nnz
    //  ratio = ratio - epsilon
    // end while
    // indices = get indices
    // values = get values
    return;
}

