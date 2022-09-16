/* Copyright 2018 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// C-only header for Realm - mostly includes typedefs right now,
//  but may be expanded to provide C bindings for the Realm API

#ifndef KITSUNE_REALM_C_H
#define KITSUNE_REALM_C_H

#include <realm/realm_c.h>
#include <realm.h>

// for size_t
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
  //typedef struct context context;
  struct context* getRealmCTX();
  int realmInitRuntime(int argc, char** argv);
  //NOTE: realmSpawn declared and defined in wrapper.cc
  //int realmSpawn();
  //int realmSync();
  size_t realmGetNumProcs();
  void realmFinalize();

#ifdef __cplusplus
}
#endif

#endif // ifndef KITSUNE_REALM_C_H
