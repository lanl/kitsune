//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// Simple example of an element-wise vector sum.
// To enable kitsune+tapir compilation add the flags to a standard
// clang compilation:
//
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/timer.h"
#include "kitrt/kitcuda/cuda.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void fill(float *data, size_t N) {
   for(size_t i = 0; i < N; ++i)
     data[i] = float(i);
}

int main (int argc, char* argv[]) {
  size_t size = VEC_SIZE;

  if (argc > 1)
    size = atol(argv[1]);

  fprintf(stdout, "problem size: %ld\n", size);

  timer r;

  float *A = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *B = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *C = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);

  random_fill(A, size);
  random_fill(B, size);

  __kitrt_cuEnableEventTiming(0);
  forall(size_t i = 0; i < size; i++)
      C[i] = A[i] + B[i];

  double time = __kitrt_cuGetLastEventTime();
  fprintf(stdout, "kernel time: %7lg\n", time);

  // Sanity check the results...  We will take a hit here on 
  // page faults back on the CPU side.  This will show up on
  // the overall runtime of the program (e.g., the forall 
  // might be faster but the overhead of page faults will 
  // wipe that out). 
  size_t error_count = 0;
  for(size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum) {
      error_count++;
    }
  }

  if (error_count > 0)
    printf("bad result!\n");
  else {
   double rtime = r.seconds();
    fprintf(stdout, "total runtime: %7lg\n", rtime);
  }

  return 0;
}
