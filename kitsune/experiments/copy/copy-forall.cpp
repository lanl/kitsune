// See the README file for details.
#include <iostream>
#include <fstream>
#include <chrono>
#include <kitsune.h>
#include <cmath>
#include "kitsune/timer.h"
#include "kitrt/kitcuda/cuda.h"

using namespace std;
using namespace kitsune;

const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;
const unsigned int DEFAULT_ITERATIONS = 10;

template <typename T>
T* alloc(int N) {
  return (T*)__kitrt_cuMemAllocManaged(sizeof(T) * N);
}

template <typename T>
void dealloc(T* array) {
  __kitrt_cuMemFree((void*)array);
}

template <typename T>
void random_fill(T* data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (T)RAND_MAX;
  // we've updated the data array -- flag it for
  // prefetching the next time we launch a kernel
  // (forall loop) that uses it...
  __kitrt_cuMemNeedsPrefetch(data);
}

template <typename T>
bool check(const T* data0, const T* data1, size_t N) {
  for(size_t i = 0; i < N; ++i) {
    if (data0[i] != data1[i])
      return false;
  }
  // In the process of the check we've paged 'data1'
  // back to the host.  Flag it as needing a prefetch
  // again; this forces all pages to be moved prior
  // to a kernel launch that is dependent upon it.
  // This highlights an issue with UVM usage as we
  // would really like a copy resident on the GPU and
  // here to be checked on the CPU...
  __kitrt_cuMemNeedsPrefetch((void *)data1);
  return true;
}

template <typename T>
void parallel_copy(T* dst, const T* src, int N) {
  forall(int i = 0; i < N; i++)
    dst[i] = src[i];
}


int main(int argc, char** argv) {
  size_t array_size = DEFAULT_ARRAY_SIZE;
  unsigned int iterations = DEFAULT_ITERATIONS;
  if (argc >= 2)
    array_size = atol(argv[1]);

  if (argc == 3)
    iterations = atoi(argv[2]);

  fprintf(stdout, "array size: %ld\n", array_size);
  fprintf(stdout, "iterations = %d\n", iterations);
  float *data0 = alloc<float>(array_size);
  float *data1 = alloc<float>(array_size);

  random_fill(data0, array_size);

  auto start = chrono::steady_clock::now();
  for(int i = 0; i < iterations; i++) {
    auto copy_start = chrono::steady_clock::now();
    parallel_copy(data1, data0, array_size);
    auto copy_end = chrono::steady_clock::now();
    cout  << "Copy time: "
          << chrono::duration<double>(copy_end-copy_start).count()
          << endl;

    if (i+1 < iterations)
      random_fill(data0, array_size);
  }

  // This will page fault GPU-side pages back to the
  // CPU -- page faults should impact our overall
  // performance.
  if (not check(data0, data1, array_size)) {
    fprintf(stderr, "final copies not equal!\n");
    return 1;
  }

  auto end = chrono::steady_clock::now();
  cout << "Total time: "
       << chrono::duration<double>(end-start).count()
       << endl;
  return 0;
}
