#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "kernels.cuh"
#include "utils.h"

void Verify(float* in, float* out_vals, int* out_indices, int out_num) {
    int acc = 0;
    float sum = 0;
    for (int i = 0; i < out_num; ++i) {
        sum += out_vals[i];
        int idx = out_indices[i];
        if (abs(in[idx]-out_vals[i]) > 0.001f) {
            printf("wrong results at i %d idx %d in the original data.\n", i, idx);
            printf("picked val: %f, original val: %f.\n", out_vals[i], in[idx]);
            if (acc++ > 10)
                break;
        }
    }
    sum = sum/out_num;
    printf("avg vals for top %d is: %f \n", out_num, sum);
}

int main()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 10.0f);
    int num_items = 1024*1024*32;
    float eta = 0.0001f;

    cub::CachingDeviceAllocator g_allocator(true);

    float* in = new float[num_items];
    float* out_vals = new float[num_items];
    int* out_indices = new int[num_items];
    int* out_num = new int[1];

    for (int i = 0; i < num_items; ++i) {
        in[i] = distribution(generator);
    }
   
    GpuTimer timer;
    for (int i = 1; i < 2; ++i) {
        int k = 1024*i;
        timer.Start();
        TrimmedTopK<1024>(g_allocator, in, out_vals, out_indices, out_num, k, eta, num_items);
        timer.Stop();
        printf("testing for k = %d\n", k);
        Verify(in, out_vals, out_indices, out_num[0]);
        printf("elapsed milliseconds: %f\n", timer.ElapsedMillis());
        printf("================================\n");
    }

    if (in) delete[] in;
    if (out_vals) delete[] out_vals;
    if (out_indices) delete[] out_indices;
    if (out_num) delete[] out_num;
}
