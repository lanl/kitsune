#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<kitsune.h>
#include<gpu.h>
#include<stdint.h>
#include<omp.h>
#include<assert.h>

reduction
void sum(int *a, int b, int unit){
  *a += b + unit;
}

int triangle(uint64_t n, int* a){
  int red = 0; 
  forall(uint64_t i=0; i<n; i++){
    sum(&red, a[i], 0); 
  }
  return red; 
}

int main(int argc, char** argv){
  int e = argc > 1 ? atoi(argv[1]) : 100; 
  int niter = argc > 2 ? atoi(argv[2]) : 100; 
  uint64_t n = e;//1ULL<<e; 
  int* arr = (int*)gpuManagedMalloc(sizeof(int) * n); 

  forall(uint64_t i=0; i<n; i++){
    arr[i] = i; 
  }

  printf("%d, %d, %d\n", triangle(n, arr), n, (n*(n-1))/2);

  int par = 0; 
  double before = omp_get_wtime();
  for(int i=0; i<niter; i++){
    par = triangle(n, arr);
    assert(par == (n*(n-1))/2);
  }
  double after = omp_get_wtime(); 
  double partime = after - before; 

  printf("%d in %f s\n" , par, partime);
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("bandwidth: %f GB/s \n" , bw);
}

