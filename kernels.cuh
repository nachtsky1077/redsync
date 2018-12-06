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
        return (abs(a) > compare);
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
        return (abs(a.y) > compare);
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
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < num_items; i+=STRIDE) {
        keys[i] = in[i].x;
        vals[i] = in[i].y;
        if (keys[i] == 16783392)
            printf("test test: %d, %f %f\n", i, in[i].x, in[i].y);
    }
}

__global__ void CreateKV(int *keys, float *vals, float2 *out, int num_items) {
    size_t STRIDE = gridDim.x * blockDim.x;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < num_items; i+=STRIDE) {
        float2 kv_pair = make_float2(keys[i], vals[i]);
        out[i] = kv_pair;
        if (i == 16783392)
            printf("test test: %d, %f %f\n", i, out[i].x, out[i].y);
    }
}

template<int NUM_THREADS>
void GetFiltered(cub::CachingDeviceAllocator &g_allocator, float *in_vals, int *in_keys, float2 *output_kvs, int *num_out, float threshold, int num_items) {
    float2 *input_kvs = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&input_kvs, sizeof(float2)*num_items)); 
    CreateKV<<<128, NUM_THREADS>>>(in_keys, in_vals, input_kvs, num_items);

    LargerThanKV select_op(threshold);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_kvs, output_kvs, num_out, num_items, select_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    printf("need %ud temp_storage_bytes.\n", temp_storage_bytes);
    CubDebugExit(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_kvs, output_kvs, num_out, num_items, select_op)); 

    if (input_kvs) CubDebugExit(g_allocator.DeviceFree(input_kvs));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

template <typename T, typename SizeT>
__global__ void MemsetIdxKernel(T *d_out, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] = idx;
    }
}

template <typename T, typename SizeT>
__global__ void MemsetKernel(T *d_out, SizeT length, T val)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] = val;
    }
}

// TODO: add multi-stream pipelining
template<int NUM_THREADS>
void TrimmedTopK(cub::CachingDeviceAllocator &g_allocator, float *in, float *out_vals, int *out_indices, int *out_num, int k, float eta, int num_items) {
    // create all device variables.
    float *d_in_vals = NULL;
    int *d_in_indices = NULL;
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

    CubDebugExit(cudaMemcpy(d_in_vals, in, num_items*sizeof(float), cudaMemcpyHostToDevice));

    // compute mean and max
    float mean[1];
    float max[1];
    MeanKernel(g_allocator, d_in_vals, d_out_sum, num_items);

    MaxKernel(g_allocator, d_in_vals, d_out_max, num_items);

    CubDebugExit(cudaMemcpy(&mean[0], d_out_sum, 1*sizeof(float), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(&max[0], d_out_max, 1*sizeof(float), cudaMemcpyDeviceToHost));

    mean[0] = mean[0] / num_items;

    float l = 0.0f;
    float r = 1.0f;
    float threshold = 0.0f;
    int nnz[1];

    while (r - l > eta) {
        float ratio = l + (r-l)/2;
        threshold = mean[0] + ratio * (max[0] - mean[0]);

        CountNonZero(g_allocator, d_in_vals, d_out_vals, d_out_num, threshold, num_items);

        CubDebugExit(cudaMemcpy(&nnz[0], d_out_num, 1*sizeof(int), cudaMemcpyDeviceToHost));
        printf("test test test nnz:%d\n", nnz[0]);

        if (nnz[0] > k && nnz[0] < 2*k) {
            break;
        }
        else {
            if (nnz[0] < k/2) {
                r = ratio;
            }
            else {
                l = ratio;
            }
        }
    }

    MemsetIdxKernel<<<128, NUM_THREADS>>>(d_in_indices, num_items);
    MemsetKernel<<<128, NUM_THREADS>>>(d_out_vals, num_items, 0.0f);

    float2 *output_kvs = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&output_kvs, sizeof(float2)*num_items));

    GetFiltered<NUM_THREADS>(g_allocator, d_in_vals, d_in_indices, output_kvs, d_out_num, threshold, num_items);

    // indices = get indices
    // values = get values
    CubDebugExit(cudaMemcpy(out_num, d_out_num, 1*sizeof(int), cudaMemcpyDeviceToHost));
    printf("final top number picked: %d final threshold: %f\n", out_num[0], threshold);
    GetKV<<<128, NUM_THREADS>>>(output_kvs, d_out_indices, d_out_vals, out_num[0]);
    CubDebugExit(cudaMemcpy(out_vals, d_out_vals, out_num[0]*sizeof(float), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(out_indices, d_out_indices, out_num[0]*sizeof(int), cudaMemcpyDeviceToHost));

    if (output_kvs) CubDebugExit(g_allocator.DeviceFree(output_kvs));
    if (d_in_vals) cudaFree(d_in_vals);
    if (d_in_indices) cudaFree(d_in_indices);
    if (d_out_max) cudaFree(d_out_max);
    if (d_out_sum) cudaFree(d_out_sum);
    if (d_out_num) cudaFree(d_out_num);
    if (d_out_vals) cudaFree(d_out_vals);
    if (d_out_indices) cudaFree(d_out_indices);

    return;
}

