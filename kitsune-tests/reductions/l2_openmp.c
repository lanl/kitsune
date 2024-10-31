#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<gpu.h>

double l2(uint64_t n, double* a){
  double red = 0; 
  #pragma omp parallel for reduction(+:red)
  for(uint64_t i=0; i<n; i++){
    red += a[i]*a[i]; 
  }

  return sqrt(red);
}

double l2_seq(uint64_t n, double* a){
  double red = 0; 
  for(uint64_t i=0; i < n; i++){
    red += a[i]*a[i]; 
  }
  return sqrt(red);
}

int main(int argc, char** argv){
  int e = argc > 1 ? atoi(argv[1]) : 28; 
  int niter = argc > 2 ? atoi(argv[2]) : 100; 
  uint64_t n = 1ULL<<e; 
  double* arr = malloc(sizeof(double) * n); 

  #pragma omp parallel for
  for(uint64_t i=0; i<n; i++){
    arr[i] = i; 
  }

  l2(n, arr);

  clock_t before = clock();
  double par; 
  for(int i=0; i<niter; i++){
    par = l2(n, arr);
  }
  clock_t after = clock(); 
  double partime = (double)(after - before) / 1000000; 

  before = clock();
  double seq = l2_seq(n, arr);
  after = clock(); 
  double seqtime = (double)(after - before) / 1000000; 

  printf("par: %f in %f s , seq: %f in %f s\n" , par, partime, seq, seqtime);
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("par bandwidth: %f GB/s \n" , bw);
}

