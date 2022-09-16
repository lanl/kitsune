#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/timer.h"
#include "kitsune/kitrt/llvm-gpu.h"
#include "kitsune/kitrt/kitrt-cuda.h"

using namespace std;
using namespace kitsune;

const size_t ARRAY_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = float(i);
}

int main (int argc, char* argv[]) {
  size_t size = ARRAY_SIZE;
  if (argc > 1)
    size = atol(argv[1]);

  fprintf(stdout, "problem size: %ld\n", size);
  float *A = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *B = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *C = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);

  random_fill(A, size);
  random_fill(B, size);  

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < size; i++)
    C[i] = A[i] + B[i];

  printf("%f\n", C[10]);
  return 0;
}


