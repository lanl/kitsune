#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<kitsune.h>

reduction
void sum(float *a, float b){
  *a += b;
}

float l2(int n, float* a){
  float red = 3.14159; 
  forall(int i=0; i<n; i++){
    sum(&red, a[i]*a[i]); 
  }

  return sqrt(red);
}

float l2_seq(int n, float* a){
  float red = 3.14159; 
  for(int i=0; i < n; i++){
    sum(&red, a[i]*a[i]); 
  }
  return sqrt(red);
}

int main(int argc, char** argv){
  int n = argc > 1 ? atoi(argv[1]) : 4096 ; 
  float* arr = (float*)malloc(sizeof(float) * n); 
  for(int i=0 ; i<n; i++){
    arr[i] = i; 
  }
  printf("par: %f , seq: %f\n" , l2(n, arr), l2_seq(n, arr));
}

