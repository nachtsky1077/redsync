#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.cuh"

void test(cub::CachingDeviceAllocator &g_allocator,
          float* in,
          float* d_in,
          int* d_in_indices,
          float* out,
          int* out_num,
          int* out_indices,
          float* d_out,
          int* d_out_num,
          int* d_out_indices,
          float* out_mean_ref,
          float* out_max_ref,
          int* out_num_ref,
          int* out_indices_ref,
          float* out_ref,
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
        printf("mean kernel passed. %f %f\n", out[0], out_mean_ref[0]);

    MaxKernel(g_allocator, d_in, d_out, num_elements);
    CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost));
    
    flag = true;
    if (abs(out[0] - out_max_ref[0]) > 0.001f)
    {
        flag = false;
        printf("max kernel result is wrong: %f %f\n", out[0], out_max_ref[0]);
    }
    if (flag)
        printf("max kernel passed. %f %f\n", out[0], out_max_ref[0]);

    CountNonZero(g_allocator, d_in, d_out, d_out_num, threshold, num_elements);
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(cudaMemcpy(out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(out_num, d_out_num, 1*sizeof(int), cudaMemcpyDeviceToHost));

    flag = true;
    if (abs(out_num[0] - out_num_ref[0]) > 0.001f)
    {
        flag = false;
        printf("count nnz kernel result is wrong: %d %d\n", out_num[0], out_num_ref[0]);
    }
    if (flag)
        printf("count nnz kernel passed. out_num: %d out_num_ref: %d\n", out_num[0], out_num_ref[0]);

    CubDebugExit(cudaMemcpy(d_in_indices, out_indices, num_elements*sizeof(int), cudaMemcpyHostToDevice));

    GetFiltered<1024>(g_allocator, d_in, d_in_indices, d_out, d_out_indices, d_out_num, threshold, num_elements);
    CubDebugExit(cudaMemcpy(out_num, d_out_num, 1*sizeof(int), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(out, d_out, out_num[0]*sizeof(float), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(out_indices, d_out_indices, out_num[0]*sizeof(int), cudaMemcpyDeviceToHost));

    flag = true;
    if (abs(out_num[0] - out_num_ref[0]) > 0.001f)
    {
        flag = false;
        printf("get filtered kernel result is wrong: %d %d\n", out_num[0], out_num_ref[0]);
    }

    for (int i = 0; i < out_num[0]; ++i) {
        if (abs(out[i] - out_ref[i]) > 0.001f) {
            flag = false;
            printf("get filtered kernel result is wrong at idx %d: %f %f", i, out[i], out_ref[i]);
            break;
        }
        if (out_indices[i] != out_indices_ref[i]) {
            flag = false;
            printf("get filtered kernel indices result is wrong at idx %d: %d %d", i, out_indices[i], out_indices_ref[i]);
            break;
        }
    }
    if (flag)
        printf("get filtered kernel passed.\n");

}

int main()
{
    srand(19850814);
    int num_elements = 512*256*24;
    
    float* data = (float*)malloc(num_elements * sizeof(float));
    float* result = (float*)malloc(num_elements * sizeof(float));
    int* result_indices = (int*)malloc(num_elements * sizeof(int));
    int* result_num = (int*)malloc(1 * sizeof(int));
    float* result_ref_max = (float*)malloc(1 * sizeof(float));
    float* result_ref_mean = (float*)malloc(1 * sizeof(float));
    float* result_ref_vals = (float*)malloc(num_elements * sizeof(float));
    int* result_ref_indices = (int*)malloc(num_elements * sizeof(int));
    int* result_ref_num = (int*)malloc(1 * sizeof(int));
    
    memset(result_ref_max, 0.0f, 1 * sizeof(float));
    memset(result_ref_mean, 0.0f, 1 * sizeof(float));
    memset(result_ref_vals, 0.0f, num_elements * sizeof(float));
    memset(result, 0.0f, 1 * sizeof(float));
    memset(result_ref_num, 0, 1 * sizeof(int));
    memset(result_num, 0, 1 * sizeof(int));
    memset(result_indices, 0, num_elements * sizeof(int));
    memset(result_ref_indices, 0, num_elements * sizeof(int));

    float* d_data = NULL;
    int* d_data_indices = NULL;
    float* d_result = NULL;
    int* d_result_num = NULL;
    int* d_result_indices = NULL;
    float threshold = 5.0f;

    cub::CachingDeviceAllocator g_allocator(true);

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(float)*num_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data_indices, sizeof(int)*num_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result, sizeof(float)*num_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_num, sizeof(int)*1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result_indices, sizeof(int)*num_elements));

    int acc = 0;
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
        result_indices[i] = i;
        if (data[i] > threshold) {
            result_ref_indices[acc] = i;
            result_ref_num[0] += 1;
            result_ref_vals[acc++] = data[i];
        }
    }
    result_ref_mean[0] = result_ref_mean[0] / num_elements;

    // Test MeanKernel
    // Test MaxKernel
    // Test CountNonZero
    test(g_allocator,
          data,
          d_data,
          d_data_indices,
          result,
          result_num,
          result_indices,
          d_result,
          d_result_num,
          d_result_indices,
          result_ref_mean,
          result_ref_max,
          result_ref_num,
          result_ref_indices,
          result_ref_vals,
          threshold,
          num_elements);

    if (data) delete[] data;
    if (result) delete[] result;
    if (result_indices) delete[] result_indices;
    if (result_ref_max) delete[] result_ref_max;
    if (result_ref_mean) delete[] result_ref_mean;
    if (result_ref_vals) delete[] result_ref_vals;
    if (result_num) delete[] result_num;
    if (result_ref_num) delete[] result_ref_num;
    if (result_ref_indices) delete[] result_ref_indices;
    if (d_data) cudaFree(d_data);
    if (d_data_indices) cudaFree(d_data_indices);
    if (d_result) cudaFree(d_result);
    if (d_result_num) cudaFree(d_result_num);
    if (d_result_indices) cudaFree(d_result_indices);
}
