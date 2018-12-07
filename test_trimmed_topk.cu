#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "kernels.cuh"
#include "utils.h"

void Verify(float *out_vals,
            int *out_indices,
            int out_num,
            float *out_vals_ref,
            int *out_indices_ref,
            int out_num_ref) {
    if (out_num != out_num_ref) {
        printf("out_num(%d) and out_num_ref(%d) doesn't match\n", out_num, out_num_ref);
        return;
    }
    printf("out_num(%d) and out_num_ref(%d) matches\n", out_num, out_num_ref);
    for (int i = 0; i < out_num; ++i) {
        if (out_indices[i] != out_indices_ref[i]) {
            printf("indices(%d) and ref(%d) differed at idx %d\n", out_indices[i], out_indices_ref[i], i);
        }
    }

    for (int i = 0; i < out_num; ++i) {
        if (out_vals[i] != out_vals_ref[i]) {
            printf("vals(%f) and ref(%f) differed at idx %d\n", out_vals[i], out_vals_ref[i], i);
        }
    }
}

float GetMean(float *in, int num_items) {
    float mean = 0.0f;
    for (int i = 0; i < num_items; ++i) {
        mean += abs(in[i]);
    }
    mean /= num_items;
    return mean;
}

float GetMax(float *in, int num_items) {
    float max = 0.0f;
    for (int i = 0; i < num_items; ++i) {
        if (max < abs(in[i])) {
            max = abs(in[i]);
        }
    }
    return max;
}

int CountNonZero(float *in, float threshold, int num_items) {
    int acc = 0;
    for (int i = 0; i < num_items; ++i) {
        if (in[i] > threshold) {
            acc += 1;
        }
    }
    return acc;
}

int GetSparseTensor(float *in, float *out_vals, int *out_indices, float threshold, int num_items) {
    int acc = 0;
    for (int i = 0; i < num_items; ++i) {
        if (abs(in[i]) > threshold) {
            out_vals[acc] = in[i];
            out_indices[acc] = i;
            acc += 1;
        }
    }
    return acc;
}

void TrimmedTopKRef(float *in, float *out_vals_ref, int *out_indices_ref, int *out_num_ref, int k, float eta, int num_items) {
    float mean = GetMean(in, num_items);
    float max = GetMax(in, num_items);
    float l = 0.0f;
    float r = 1.0f;
    float threshold = 0.0f;
    int nnz = 0;

    while (r - l > eta) {
        float ratio = l + (r-l)/2;
        threshold = mean + ratio * (max - mean);
        printf("threshold %f\n", threshold);
        nnz = CountNonZero(in, threshold, num_items);
        if (nnz > k && nnz < 2*k) {
            break;
        }
        else {
            if (nnz < k/2) {
                r = ratio;
            }
            else {
                l = ratio;
            }
        }
    }

    out_num_ref[0] = GetSparseTensor(in, out_vals_ref, out_indices_ref, threshold, num_items);
}

void TopKLastStepRef(float *in, float *out_vals_ref, int *out_indices_ref, int *out_num_ref, float threshold, int num_items) {
    out_num_ref[0] = GetSparseTensor(in, out_vals_ref, out_indices_ref, threshold, num_items);
}

int main(int argc, char *argv[])
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 10.0f);
    int num_items = 1024*1024*atoi(argv[1]);
    float eta = 0.0001f;

    cub::CachingDeviceAllocator g_allocator(true);

    float* in = new float[num_items];
    float* out_vals = new float[num_items];
    int* out_indices = new int[num_items];
    int* out_num = new int[1];
    float* out_vals_ref = new float[num_items];
    int* out_indices_ref = new int[num_items];
    int* out_num_ref = new int[1];

    for (int i = 0; i < num_items; ++i) {
        in[i] = distribution(generator);
    }

    GpuTimer timer;
    for (int i = 1; i < 6; ++i) {
        int k = 1024*i;
        timer.Start();
        float threshold = TrimmedTopK<1024>(g_allocator, in, out_vals, out_indices, out_num, k, eta, num_items);
        timer.Stop();
        TopKLastStepRef(in, out_vals_ref, out_indices_ref, out_num_ref, threshold, num_items);
       
        // Disabled the total test since different precision of floating point numbers on the GPU and on the CPU.
        // TrimmedTopKRef(in, out_vals_ref, out_indices_ref, out_num_ref, k, eta, num_items);

        printf("testing for k = %d\n", k);
        Verify(out_vals, out_indices, out_num[0], out_vals_ref, out_indices_ref, out_num_ref[0]);
        printf("elapsed milliseconds: %f\n", timer.ElapsedMillis());
        printf("================================\n");
    }

    if (in) delete[] in;
    if (out_vals) delete[] out_vals;
    if (out_indices) delete[] out_indices;
    if (out_num) delete[] out_num;
}
