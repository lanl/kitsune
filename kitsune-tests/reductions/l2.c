#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<kitsune.h>
#include<omp.h>
#include<gpu.h>

reduction
void sum(double *a, double b){
  *a += b;
}

__attribute__((noinline))
double l2(uint64_t n, double* a){
  double red = 0; 
  forall(uint64_t i=0; i<n; i++){
    sum(&red, x); 
  }

  return sqrt(red); 
}

int main(int argc, char** argv){
  int e = argc > 1 ? atoi(argv[1]) : 28; 
  int niter = argc > 2 ? atoi(argv[2]) : 100; 
  uint64_t n = 1ULL<<e; 
  double* arr = (double*)gpuManagedMalloc(sizeof(double) * n); 

  forall(uint64_t i=0; i<n; i++){
    arr[i] = (double)i; 
  }

  printf("result: %f \n", l2(n, arr));

  double res[niter];
  double before = omp_get_wtime(); 
  for(int i=0; i<niter; i++){
    double red = l2(n, arr); 
    //printf("red: %f\n", red); 
    res[i]=red; 
  }
  double after = omp_get_wtime(); 

  double partime = (double)(after - before); 
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("bandwidth: %f GB/s \n" , bw);

  //double time = (double)(after - before) / 1000000; 
  //double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * time);  
  //printf("bandwidth: %f GB/s \n" , bw);
}

