#include"magma.h"
#include<assert.h>
#include<unistd.h>
#include<stdio.h>
//#include<kitsune.h>

template <typename a, typename um, typename v> 
a reduce(um m, v& xs){
  auto acc = m.id(); 
  for(auto x : xs){
    acc = m.op(acc, x); 
  }
  return acc; 
}

template <typename a, typename um, typename v>
a parReduce(um m, v& xs, uint64_t nthreads){
  uint64_t linesize = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); 
  assert(linesize % sizeof(a) == 0); 
  uint64_t linenum = linesize / sizeof(a); 
  a* accs = new a[nthreads * linenum]; 
  uint64_t size = xs.end() - xs.begin(); 
  assert(size % nthreads == 0); 
  uint64_t grainsize = size / nthreads; 
  for(uint64_t i=0; i<nthreads; i++){
    accs[i*linenum] = m.id();
    for(uint64_t j = i*grainsize; j<(i+1)*grainsize; j++){
      accs[i*linenum] = m.op(accs[i*linenum], xs[j]);
    }
  }
  a acc = m.id(); 
  for(uint64_t i=0; i<nthreads; i++){
    acc = m.op(acc, accs[i*linenum]); 
  }
  delete[] accs; 
  return acc; 
}

template<typename a, typename um, typename v> 
a treeReduce(um m, v& xs, uint64_t start, uint64_t end, uint64_t gs){
  if(end-start < gs){
    a acc = m.id();
    for(uint64_t i=start; i<end; i++){
      acc+=xs[i];
    }
    return acc; 
  }
  else{
    uint64_t mid = (start + end) / 2; 
    return treeReduce(m, xs, start, mid, gs) 
         + treeReduce(m, xs, mid, end, gs); 
  }
}

  

