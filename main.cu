#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.cuh"

void test(cub::CachingDeviceAllocator &g_allocator,
          float* in,
          float* d_in,
          float* out,
          int* out_num,
          float* d_out,
          int* d_out_num,
          float* out_mean_ref,
          float* out_max_ref,
          int* out_num_ref,
          float threshold,
          int num_elements)
{ 
    CubDebugExit(cudaMemcpy(d_in, in, num_elements*sizeof(float), cudaMemcpyHostToDevice));
    MeanKernel(g_allocator, d_in, d_out, num_elements);
    CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost));

    bool flag = true;
    out[0] = out[0] / num_elements;
    if (abs(out[0] - out_mean_ref[0]) > 0.001f)
    {
        flag = false;
        printf("mean kernel result is wrong: %f %f\n", out[0], out_mean_ref[0]);
    }
    if (flag)
        printf("mean kernel passed.\n");

    MaxKernel(g_allocator, d_in, d_out, num_elements);
    CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost));
    
    flag = true;
    if (abs(out[0] - out_max_ref[0]) > 0.001f)
    {
        flag = false;
        printf("max kernel result is wrong: %f %f\n", out[0], out_max_ref[0]);
    }
    if (flag)
        printf("max kernel passed.\n");

    CountNonZero(g_allocator, d_in, d_out, d_out_num, threshold, num_elements);
    CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(out_num, d_out_num, 1*sizeof(int), cudaMemcpyDeviceToHost));

    flag = true;
    if (abs(out_num[0] - out_num_ref[0]) > 0.001f)
    {
        flag = false;
        printf("count nnz kernel result is wrong: %d %d\n", out_num[0], out_num_ref[0]);
    }
    if (flag)
        printf("max kernel passed.\n");
}

int main()
{
    srand(19850814);
    int num_elements = 512*256*24;
    
    float* data = (float*)malloc(num_elements * sizeof(float));
    float* result = (float*)malloc(1 * sizeof(float));
    int* result_num = (int*)malloc(1 * sizeof(int));
    float* result_ref_max = (float*)malloc(1 * sizeof(float));
    float* result_ref_mean = (float*)malloc(1 * sizeof(float));
    int* result_ref_num = (int*)malloc(1 * sizeof(int));
    
    memset(result_ref_max, -1.0f, 1 * sizeof(float));
    memset(result_ref_mean, 0.0f, 1 * sizeof(float));
    memset(result, 0.0f, 1 * sizeof(float));
    memset(result_ref_num, 0, 1 * sizeof(int));
    memset(result_num, 0, 1 * sizeof(int));

    float* d_data = NULL;
    float* d_result = NULL;
    int* d_result_num = NULL;
    float threshold = 5.0f;

    cub::CachingDeviceAllocator g_allocator(true);

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(float)*num_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result, sizeof(float)*num_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_num, sizeof(int)*1));

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
    test(g_allocator,
          data,
          d_data,
          result,
          result_num,
          d_data,
          d_result_num,
          result_ref_mean,
          result_ref_max,
          result_ref_num,
          threshold,
          num_elements);

    if (data) delete[] data;
    if (result) delete[] result;
    if (result_ref_max) delete[] result_ref_max;
    if (result_ref_mean) delete[] result_ref_mean;
    if (result_num) delete[] result_num;
    if (result_ref_num) delete[] result_ref_num;
    if (d_data) cudaFree(d_data);
    if (d_result) cudaFree(d_result);
    if (d_result_num) cudaFree(d_result_num);
}
