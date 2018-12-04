#define CUB_STDERR

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
    CUB_RUNTIME_FUNCTION __forceinline__
    LessThan(float compare) : compare(compare) {}

    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const float &a) const {
        return (a > compare);
    }
};


void MeanKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items);

void MaxKernel(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int num_items);

void CountNonZero(cub::CachingDeviceAllocator &g_allocator, float *in, float *out, int *num_out, float threshold, int num_items);

