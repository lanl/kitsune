#include<stdlib.h>
#include<stdio.h>

extern void matmul(float* ao, float *a, uint64_t a1, uint64_t a2, uint64_t a3, 
									 float* bo, float *b, uint64_t b1, uint64_t b2, uint64_t b3, 
									 float* co, float *c, uint64_t c1, uint64_t c2, uint64_t c3, 
									 uint64_t n, uint64_t k, uint64_t m); 
				

int main(int argc, char** argv){
	int n = argc > 1 ? atoi(argv[1]) : 256;
	int k = argc > 2 ? atoi(argv[2]) : 256;
	float* a = malloc(sizeof(float) * n * n);
	float* b = malloc(sizeof(float) * n * n);
	float* c = malloc(sizeof(float) * n * n);
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++){
			a[i*n+j] = ((float)rand())/RAND_MAX;
			b[i*n+j] = ((float)rand())/RAND_MAX;
		}

	for(int i=0; i<k; i++){
		matmul(NULL, a, 0, 0, 0, 
					 NULL, b, 0, 0, 0,
					 NULL, c, 0, 0, 0,
					 n, n, n);
	}
}
		
