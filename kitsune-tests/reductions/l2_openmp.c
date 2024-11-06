#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<gpu.h>
#include<omp.h>

double l2(uint64_t n, double* a){
  double red = 0; 
  #pragma omp parallel for reduction(+:red)
  for(uint64_t i=0; i<n; i++){
    red += a[i]*a[i]; 
  }

  return sqrt(red);
}

double l2_grid(uint64_t n, double* a){
  double red[12];
  #pragma omp parallel for
  for(uint64_t i=0; i<12; i++){
    red[i] = 0; 
    for(uint64_t j=i; j<n; j+=12){
      red[i] += a[j]*a[j]; 
    }
  }

  double res = 0; 
  for(int i=0; i<12; i++) res += red[i]; 
  return sqrt(res); 
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

  double before = omp_get_wtime(); 
  double par; 
  for(int i=0; i<niter; i++){
    par = l2(n, arr);
  }
  double after = omp_get_wtime(); 
  double partime = (double)(after - before); 
  printf("par: %f in %f s\n" , par, partime);
  double bw = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime);  
  printf("par bandwidth: %f GB/s \n" , bw);

  double before_grid = omp_get_wtime();
  double grid; 
  for(int i=0; i<niter; i++){
    grid = l2_grid(n, arr);
  }
  double after_grid = omp_get_wtime(); 
  double partime_grid = (double)(after_grid - before_grid); 
  printf("grid: %f in %f s\n" , grid, partime_grid);
  double bw_grid = (double)((1ULL<<e) * niter * sizeof(double)) / (1000000000.0 * partime_grid);  
  printf("grid bandwidth: %f GB/s \n" , bw_grid);
}

