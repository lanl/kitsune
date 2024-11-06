#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<kitsune.h>
#include<gpu.h>

reduction
void sum(double *a, double b){
  *a += b;
}

__attribute__((noinline))
double l2(uint64_t n, double* a){
  double red = 0; 
  forall(uint64_t i=0; i<n; i++){
    sum(&red, a[i]*a[i]); 
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

  printf("par:%f \n", l2(n, arr));

  /*
  clock_t before = clock();
  double par; 
  for(int i=0; i<niter; i++){
    par = l2(n, arr);
  }
  clock_t after = clock(); 
  double partime = (double)(after - before) / 1000000; 

  printf("par: %f in %f s \n" , par, partime);
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("par bandwidth: %f GB/s \n" , bw);
  */
}

