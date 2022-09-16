
#include <cuda_runtime.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "kitsune/timer.h"

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for (size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

__global__ void VectorAdd(float *A, float *B, float *C, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}


int main(int argc, char *argv[]) {
  size_t size = VEC_SIZE;
  if (argc > 1 )
    size = atol(argv[1]);

  fprintf(stdout, "problem size: %ld\n", size);

  kitsune::timer r;

  // This is loosely for consistency with the launch parameters
  // from kitsune.
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEventRecord(start);
  cudaError_t err = cudaSuccess;
  float *A, *B, *C;
  err = cudaMallocManaged(&A, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for A!\n");
    return 1;
  }
  err = cudaMallocManaged(&B, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for B!\n");
    return 1;
  }
  err = cudaMallocManaged(&C, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for C!\n");
    return 1;
  }

  random_fill(A, size);
  random_fill(B, size);
  cudaEvent_t kstart, kstop;
  cudaEventCreate(&kstart);
  cudaEventCreate(&kstop);
  cudaEventRecord(kstart);
  VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
  cudaEventRecord(kstop);
  cudaEventSynchronize(kstop);

  float msecs = 0;
  cudaEventElapsedTime(&msecs, kstart, kstop);
  printf("kernel time: %7lg\n", msecs / 1000.0);

  // Sanity check the results...
  size_t error_count = 0;
  for (size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count != 0)
    printf("bad result!\n");
  else {
    double rtime = r.seconds();
    fprintf(stdout, "total runtime: %7lg\n", rtime);
  }

  return 0;
}
