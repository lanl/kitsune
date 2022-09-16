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
//    * -fkokkos : enable specialized Kokkos recognition and
//                 compilation (lower to Tapir).
//    * -fkokkos-no-init : disable Kokkos initialization and
//                 finalization calls to avoid conflicts with
//                 target runtime operation.
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "kitsune/timer.h"
#include <cstdio>


using namespace std;
using namespace kitsune;

typedef Kokkos::DualView<float*, Kokkos::LayoutRight,
                         Kokkos::DefaultExecutionSpace>
    DualViewVector;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(DualViewVector &data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data.h_view(i) = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {
  size_t size = VEC_SIZE;
  if (argc > 1)
    size = atol(argv[1]);

  fprintf(stdout, "problem size: %ld\n", size);

  Kokkos::initialize(argc, argv); {
    timer r;
    DualViewVector A = DualViewVector("A", size);
    DualViewVector B = DualViewVector("B", size);
    DualViewVector C = DualViewVector("C", size);

    random_fill(A, size);
    A.modify_host();

    random_fill(B, size);
    B.modify_host();

    A.sync_device();
    B.sync_device();
    C.sync_device();
    C.modify_device();
    timer t;
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int i) {
      C.d_view(i) = A.d_view(i) + B.d_view(i);
      }
    );
    Kokkos::fence();
    double loop_secs = t.seconds();
    fprintf(stdout, "kernel runtime: %7lg\n", loop_secs);
    C.sync_host();
    A.sync_host();
    B.sync_host();

    // Sanity check the results...
    size_t error_count = 0;
    for (size_t i = 0; i < size; i++) {
      float sum = A.h_view(i) + B.h_view(i);
      if (C.h_view(i) != sum)
        error_count++;
    }

    if (error_count > 0)
      printf("bad result!\n");
    else {
      double rtime = r.seconds();
      fprintf(stdout, "total runtime: %7lg\n", rtime);
    }
  }

  Kokkos::finalize();

  return 0;
}

