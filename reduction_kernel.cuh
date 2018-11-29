#define CUB_STDERR

#include <cstdio>
#include <iostream>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

using namespace cub;

bool g_verbose = false;
int g_timing_iterations = 100;
int g_grid_size = 1;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Reduction(int* d_in, int* d_out)
{
    typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceT;

    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int data[ITEMS_PER_THREAD];
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);

    int aggregate = BlockReduceT(temp_storage).Sum(data);

    if (threadIdx.x == 0)
    {
        *d_out = aggregate;
    }
}

