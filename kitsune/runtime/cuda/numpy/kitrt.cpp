//===- numpy-hijack.cpp - Kitsune runtime NumPy allocator ------------------------===//
//
// TODO:
//     - Need to update LANL/Triad Copyright notice.
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// From the NumPy documentation:
//
//    Users may wish to override the internal data memory routines
//    with ones of their own. Since NumPy does not use the Python
//    domain strategy to manage data memory, it provides an
//    alternative set of C-APIs to change memory routines.
//
//
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../kitcuda.h"
#include "../../memory_map.h"
#include <numpy/arrayobject.h>

typedef struct {
  void *(*malloc)(size_t);
  void *(*calloc)(size_t, size_t);
  void *(*realloc)(void *, size_t);
  void (* free)(void *);
} KitRTAllocatorFuncs;

__attribute__((malloc))
static void *__kitrt_NumPyMalloc(void *ctx, size_t size) {
  assert(ctx != NULL && 
         "kitrt: unexpected null context pointer in numpy allocator!");
  assert(size == 0 && "kitrt: zero-sized allocation request!");
  KitRTAllocatorFuncs *funcs = (KitRTAllocatorFuncs *)ctx;
  assert(funcs != NULL && "kitrt: unexpected null function block pointer!");
  return funcs->malloc(size);
}

__attribute__((malloc))
static void *__kitrt_NumPyCalloc(void *ctx, size_t nelem, size_t elsize) {
  assert(ctx != NULL && 
         "kitrt: unexpected null context pointer in numpy allocator!");
  assert(nelem == 0 && "kitrt: zero-sized allocation request!");
  assert(elsize == 0 && "kitrt: zero-sized element request!");
  KitRTAllocatorFuncs *funcs = (KitRTAllocatorFuncs *)ctx;
  assert(funcs != NULL && "kitrt: unexpected null function block pointer!");
  return funcs->calloc(nelem, elsize);
}

__attribute__((malloc))
static void *__kitrt_NumPyRealloc(void *ctx, void *ptr, size_t new_size) {
  assert(ctx != NULL && 
         "kitrt: unexpected nulll context pointer in numpy allocator!");
  assert(ptr != NULL &&
         "kitrt: unexpected null data pointer!");
  assert(size == 0 && "kitrt: zero-sized reallocation request!");
  KitRTAllocatorFuncs *funcs = (KitRTAllocatorFuncs *)ctx;
  assert(funcs != NULL && "kitrt: unexpected null function block pointer!");
  return funcs->realloc(ptr, new_size);
}

static void __kitrt_NumPyFree(void *ctx, void *ptr, size_t size) {
  assert(ctx != NULL &&  
         "kitrt: unexpected null context pointer in numpy free");
  assert(ptr != NULL && "kitrt: unexpected null data pointer!");
  KitRTAllocatorFuncs *funcs = (KitRTAllocatorFuncs *)ctx;
  assert(funcs != NULL && "kitrt: unexpected null function block pointer!");
  funcs->free(ptr);
}

static KitRTAllocatorFuncs __kitrt_cuda_allocators_ctx = {
  __kitcuda_mem_alloc_managed,
  __kitcuda_mem_calloc_managed,
  __kitcuda_mem_realloc_managed,
  __kitcuda_mem_free
};

static PyDataMem_Handler __kitrt_data_handler = {
  "kit_rt_data_allocator",
  1,
  {
    &__kitrt_cuda_allocators_ctx,
    __kitrt_NumPyMalloc,
    __kitrt_NumPyCalloc,
    __kitrt_NumPyRealloc,
    __kitrt_NumPyFree
  }
};

static PyObject *kitrt_InfoMethod(PyObject*, PyObject*) {
  __kitrt_print_memory_map();
  Py_RETURN_NONE;
}

static PyObject *kitrt_EnableMemHandler(PyObject*, PyObject*) {
  PyObject *kitrt_handler = PyCapsule_New(&__kitrt_data_handler, "mem_handler", NULL);
  if (kitrt_handler != NULL) {
    (void)PyDataMem_SetHandler(kitrt_handler);
    Py_DECREF(kitrt_handler);
  }
  return kitrt_handler;
}

static PyObject *kitrt_DisableMemHandler(PyObject*, PyObject*) {
  (void)PyDataMem_SetHandler(NULL);
  return NULL;
}

static PyMethodDef m_methods[] = {
  {"info", kitrt_InfoMethod, METH_NOARGS, "Python interface for kitsune runtime memory map information."},
  {"disable", kitrt_DisableMemHandler, METH_NOARGS, "Disable the Kitsune runtime memory handler."},
  {"enable", kitrt_EnableMemHandler, METH_NOARGS, "Enable the Kitsune runtime memory handler."},
  {NULL, NULL, 0, NULL},
};

static PyModuleDef_Slot m_slots[] = {
  {0, NULL},
};

static PyModuleDef def = {
  PyModuleDef_HEAD_INIT,
  .m_name = "kitrt_numpy_allocator",
  .m_methods = m_methods,
  .m_slots = m_slots,
};


extern "C" PyMODINIT_FUNC PyInit_kitrt(void) {
  import_array();
  
  PyObject *kitrt_handler = PyCapsule_New(&__kitrt_data_handler, "mem_handler", NULL);
  if (kitrt_handler != NULL) {
    (void)PyDataMem_SetHandler(kitrt_handler);
    Py_DECREF(kitrt_handler);
  }
  
  return PyModuleDef_Init(&def);
}
