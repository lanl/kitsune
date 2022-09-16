#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <sys/stat.h>

#include <cuda.h>

void *fatbin = nullptr;

const char *FATBIN_FILE = "vadd.fatbin";

#define CU_SAFE_CALL(x)                                                        \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      fprintf(stderr, "kitrt: %s failed with error %s\n", #x, msg);            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void *readFatBinary(const char *filename) {
  void *fb_data = nullptr;
  struct stat st;
  stat(filename, &st);
  size_t fatbin_size = st.st_size;
  fb_data = (void *)malloc(sizeof(unsigned char) * fatbin_size);

  FILE *fp = fopen(FATBIN_FILE, "rb");
  assert(fp && "failed to open fat binary file!");
  if (fread(fb_data, sizeof(unsigned char), fatbin_size, fp) != fatbin_size) {
    fprintf(stderr, "error reading fat binary file!\n");
    free(fb_data);
    return nullptr;
  }
  fclose(fp);
  return fb_data;
}

int  main(int argc, char *argv[]) {
  fatbin = readFatBinary(FATBIN_FILE);
  if (! fatbin)
    return 1;

  /* initialize cuda */
  CUcontext context;
  CUdevice  device;
  CUstream  stream;
  CU_SAFE_CALL(cuInit(0));
  CU_SAFE_CALL(cuDeviceGet(&device, 0));
  CU_SAFE_CALL(cuCtxCreate(&context, 0, device));

  /* attempt to load the fatbinary */
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData(&module, fatbin));

  CUfunction kfunc;
  CU_SAFE_CALL(cuModuleGetFunction(&kfunc, module, "VecAdd_kernel"));

  const size_t N = 1024;
  CUdeviceptr a_data, b_data, c_data;
  CU_SAFE_CALL(cuMemAllocManaged(&a_data, sizeof(float) * N, CU_MEM_ATTACH_HOST));
  CU_SAFE_CALL(cuMemAllocManaged(&b_data, sizeof(float) * N, CU_MEM_ATTACH_HOST));
  CU_SAFE_CALL(cuMemAllocManaged(&c_data, sizeof(float) * N, CU_MEM_ATTACH_HOST));
  float *a, *b, *c;
  a = (float*)a_data;
  b = (float*)b_data;
  c = (float*)c_data;

  for(size_t i = 0; i < N; i++) {
    a[i] = float(i);
    b[i] = a[i] + 1.0f;
    c[i] = 0.0f;
  }

  CU_SAFE_CALL(cuStreamCreate(&stream, 0));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  void *args[4] = {&a, &b, &c, (void *)&N};
  CU_SAFE_CALL(cuLaunchKernel(kfunc,
                              blocksPerGrid, 1, 1,
                              threadsPerBlock, 1, 1,
                              0, stream, args, NULL));

  CU_SAFE_CALL(cuStreamSynchronize(stream));

  unsigned error_count = 0;
  for(size_t i = 0; i < N; ++i) {
    float foo = a[i] + b[i];
    if (foo != c[i]) {
      error_count++;
    }
  }
  printf("error count = %d\n", error_count);
  return 0;
}
