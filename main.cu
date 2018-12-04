#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.cuh"

bool test(cub::CachingDeviceAllocator &g_allocator,
          float* in,
          float* d_in,
          float* out,
          float* d_out,
          int* d_out_num,
          float* out_mean_ref,
          float* out_max_ref,
          float* out_num_ref,
          int num_elements)
{ 
    cub::CubDebugExit(cudaMemcpy(d_in, in, num_elements*sizeof(int), cudaMemcpyHostToDevice));
    int num_thread = 1024;
    int num_block = (num_elements + num_thread - 1) / num_thread;
    dim3 dimGrid(num_block, 1, 1);
    dim3 dimBlock(num_thread, 1, 1);
    
    cub::CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(int), cudaMemcpyDeviceToHost));

    MeanKernel(g_allocator, in, out, num_elements);

    bool flag = true;
    if (abs(out[0] - out_mean_ref[0]) < 0.001f)
    {
        flag = false;
        printf("mean kernel result is wrong: %d %d\n", out[0], out_mean_ref[0]);
    }
    if (flag)
        printf("mean kernel passed.\n");

    MaxKernel(g_allocator, in, out, num_elements);
    
    flag = true;
    if (abs(out[0] - out_max_ref[0]) < 0.001f)
    {
        flag = false;
        printf("max kernel result is wrong: %d %d\n", out[0], out_max_ref[0]);
    }
    if (flag)
        printf("max kernel passed.\n");

    CountNonZero(g_allocator, in, out, num_out, float threshold, int num_items);



    return 0;
}

int main()
{
    srand(19850814);
    int num_elements = 512*256*24;
    
    float* data = (float*)malloc(num_elements * sizeof(float));
    float* result_max = (float*)malloc(1 * sizeof(float));
    float* result_mean = (float*)malloc(1 * sizeof(float));
    int* result_num = (int*)malloc(1 * sizeof(int));
    float* result_ref_max = (float*)malloc(1 * sizeof(float));
    float* result_ref_mean = (float*)malloc(1 * sizeof(float));
    int* result_ref_num = (int*)malloc(1 * sizeof(int));
    
    memset(result_ref_max, -1.0f, 1 * sizeof(float));
    memset(result_max, 0.0f, 1 * sizeof(float));
    memset(result_ref_mean, 0.0f, 1 * sizeof(float));
    memset(result_mean, 0.0f, 1 * sizeof(float));
    memset(result_ref_num, 0, 1 * sizeof(int));
    memset(result_num, 0, 1 * sizeof(int));

    float* d_data = NULL;
    float* d_result_max = NULL;
    float* d_result_mean = NULL;
    int* d_result_num = NULL;
    float threshold = 5.0f;

    cub::CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(float)*num_elements));
    cub::CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_max, sizeof(float)*1));
    cub::CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_mean, sizeof(float)*1));
    cub::CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_num, sizeof(int)*1));

    for (int i = 0; i < num_elements; ++i)
    {
        data[i] = rand()%10*1.0f;
        if (i == 512) {
            data[i] = 13;
        }
        result_ref_mean[0] += data[i];
        if (result_ref_max[0] < data[i]) {
            result_ref_max[0] = data[i];
        }
        if (data[i] > threshold) {
            result_ref_num[0] += 1;
        }
    }
    result_ref_mean[0] = result_ref_mean[0] / num_elements;

    // Test MeanKernel

    // Test MaxKernel

    // Test CountNonZero

    if (data) delete[] data;
    if (result_max) delete[] result_max;
    if (result_ref_max) delete[] result_ref_max;
    if (result_mean) delete[] result_mean;
    if (result_ref_mean) delete[] result_ref_mean;
    if (result_num) delete[] result_num;
    if (result_ref_num) delete[] result_ref_num;
    if (d_data) cudaFree(d_data);
    if (d_result_max) cudaFree(d_result_max);
    if (d_result_mean) cudaFree(d_result_mean);
    if (d_result_num) cudaFree(d_result_num);
}
