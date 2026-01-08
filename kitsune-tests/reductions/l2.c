#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<kitsune.h>
#include<gpu.h>
#include<stdint.h>
#include<assert.h>
#include<omp.h>

reduction
void sum(double *a, double b, double unit){
  *a += b;
}

double l2(uint64_t n, double* a){
  double red = 0; 
  forall(uint64_t i=0; i<n; i++){
    sum(&red, a[i] * a[i], 0.0); 
  }

  return sqrt(red);
}

int main(int argc, char** argv){
  int e = argc > 1 ? atoi(argv[1]) : 28; 
  int niter = argc > 2 ? atoi(argv[2]) : 100; 
  uint64_t n = 1ULL<<e; 
  double* arr = (double*)gpuManagedMalloc(sizeof(double) * n); 

  forall(uint64_t i=0; i<n; i++){
    arr[i] = i; 
  }

  l2(n, arr);

  double par = 0; 
  double before = omp_get_wtime();
  for(int i=0; i<niter; i++){
    par = l2(n, arr);
  }
  double after = omp_get_wtime(); 
  double partime = after - before; 

  printf("%f in %f s\n" , par, partime);
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("bandwidth: %f GB/s \n" , bw);
}

