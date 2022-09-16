#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace std;

#include "Kokkos_DualView.hpp"

template <typename T>
using View = Kokkos::DualView<T*>;


const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;
const unsigned int DEFAULT_ITERATIONS = 10;

#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

template <typename T>
void parallel_copy(View<T> &dst, View<T> &src, int N) {
  src.sync_device();
  dst.sync_device();
  Kokkos::parallel_for("copy", N, KOKKOS_LAMBDA(const int &i) {
    dst.d_view(i) = src.d_view(i);
  });
  Kokkos::fence();
  dst.modify_device();
}

template <typename T>
void random_fill(View<T> &data, size_t N) {
  for(size_t i = 0; i < N; i++)
    data.h_view(i) = rand() / (T)RAND_MAX;
  data.modify_host();
}

template <typename T>
bool check(View<T> &data0, View<T> &data1, size_t N) {
  data0.sync_host();
  data1.sync_host();
  for(size_t i = 0; i < N; ++i) {
    if (data0.h_view(i) != data1.h_view(i))
      return false;
  }
  return true;
}


int main(int argc, char** argv)
{
  size_t array_size = DEFAULT_ARRAY_SIZE;
  unsigned int iterations = DEFAULT_ITERATIONS;
  if (argc >= 2)
    array_size = atol(argv[1]);

  if (argc == 3)
    iterations = atoi(argv[2]);

  fprintf(stderr, "array size: %ld\n", array_size);
  fprintf(stderr, "iterations = %d\n", iterations);

  auto start = chrono::steady_clock::now();
  Kokkos::initialize(argc, argv); {
    View<float> data0("data0", array_size);
    View<float> data1("data1", array_size);

    random_fill(data0, array_size);
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

    if (not check(data0, data1, array_size)) {
      fprintf(stderr, "copies not equal!\n");
      return 1;
    }

    auto end = chrono::steady_clock::now();
    cout << "Total time: " << chrono::duration<double>(end - start).count()
         << endl;

  }  Kokkos::finalize();

  return 0;
}

