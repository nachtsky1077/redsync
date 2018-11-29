#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "reduction_kernel.cuh"

template<
    int     BLOCK_THREADS,
    int     ITEMS_PER_THREADS>
bool test(int* in, int* d_in, int* out, int* d_out, int* out_ref, int num_elements)
{ 
    cudaMemcpy(d_in, in, num_elements*sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(BLOCK_THREADS, 1, 1);
    Reduction<BLOCK_THREADS, ITEMS_PER_THREADS> <<<dimGrid, dimBlock>>>(d_in, d_out);
    cudaMemcpy(out, d_out, 1*sizeof(int), cudaMemcpyDeviceToHost);

    bool flag = true;
    if (out[0] != out_ref[0])
    {
        flag = false;
        printf("result is wrong: %d %d\n", out[0], out_ref[0]);
    }
    if (flag)
        printf("passed.\n");

    return 0;
}

int main()
{
    srand(19850814);
    int num_elements = 1024;
    int* data = (int*)malloc(num_elements * sizeof(int));
    int* result = (int*)malloc(1 * sizeof(int));
    int* result_ref = (int*)malloc(1 * sizeof(int));
    memset(result_ref, 0, 1 * sizeof(int));
    memset(result, 0, 1 * sizeof(int));
    int* d_data = NULL;
    int* d_result = NULL;
    cudaMalloc((void**)&d_data, sizeof(int)*num_elements);
    cudaMalloc((void**)&d_result, sizeof(int)*1);

    for (int i = 0; i < num_elements; ++i)
    {
        data[i] = rand()%num_elements;
        result_ref[0] += data[i];
    }
    printf("testing 1024, 1: \n");
    test<1024, 1>(data, d_data, result, d_result, result_ref, num_elements);
    printf("testing 512, 2: \n");
    test<512, 2>(data, d_data, result, d_result, result_ref, num_elements);

    if (data) delete[] data;
    if (result) delete[] result;
    if (result_ref) delete[] result_ref;
    if (d_data) cudaFree(d_data);
    if (d_result) cudaFree(d_result);
}
