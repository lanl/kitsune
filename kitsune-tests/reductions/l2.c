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

double l2(uint64_t n, double* a){
  double red = 0; 
  forall(uint64_t i=0; i<n; i++){
    sum(&red, a[i]*a[i]); 
  }

  return sqrt(red);
}

double l2_seq(uint64_t n, double* a){
  double red = 0; 
  for(uint64_t i=0; i < n; i++){
    sum(&red, a[i]*a[i]); 
  }
  return sqrt(red);
}

int main(int argc, char** argv){
  uint64_t n = argc > 1 ? atoi(argv[1]) : 2ULL<<28 ; 
  double* arr = (double*)gpuManagedMalloc(sizeof(double) * n); 

  forall(uint64_t i=0; i<n; i++){
    arr[i] = i; 
  }

//  l2(n, arr);

  clock_t before = clock();
  double par = l2(n, arr);
  clock_t after = clock(); 
  double partime = (double)(after - before) / 1000000; 

  before = clock();
  double seq = l2_seq(n, arr);
  after = clock(); 
  double seqtime = (double)(after - before) / 1000000; 

  printf("par: %f in %f s , seq: %f in %f s\n" , par, partime, seq, seqtime);
  double bw = (double)((2ULL<<28) * sizeof(double)) / (1000000000.0 * partime);  
  printf("par bandwidth: %f GB/s \n" , bw);
}

