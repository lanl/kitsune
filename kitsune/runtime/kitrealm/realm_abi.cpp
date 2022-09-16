// Written by Alexis Perry-Holby for use with Tapir-LLVM

#include <realm.h>
#include <realm/threads.h>
#include <set>
#include <vector>
#include <atomic>
#include <iostream>
#include <stdlib.h>
#include <string.h>

extern "C" {

  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    std::atomic<unsigned> cur_task;
  } context;

  static context *_globalCTX;  //global variable
  static bool initCalled = false; //global variable

  context * getRealmCTX() {
    //DEBUG std::cout << "  getRealmCTX" << std::endl;
    if ( _globalCTX) 
      return _globalCTX;
    else
      return NULL;
  }
  
  Realm::Barrier createRealmBarrier(){
    return Realm::Barrier::create_barrier(1);
  }

  void destroyRealmBarrier(Realm::Barrier& b){
    b.destroy_barrier();  
  }

  
  int realmInitRuntime(int argc, char** argv) {
    //DEBUG std::cout << "realmInitRuntime" << std::endl;
    if (initCalled)
      return 0;

    _globalCTX = new context();
    assert(_globalCTX);

    _globalCTX->rt.init(&argc, &argv);

    //get CPU processors only, GPUs might be TOC_PROC instead
    Realm::Machine::ProcessorQuery procquery(Realm::Machine::get_machine());
    Realm::Machine::ProcessorQuery locprocquery = procquery.only_kind(Realm::Processor::LOC_PROC);

    _globalCTX->numprocs = locprocquery.count();
    assert (_globalCTX->numprocs > 0); //assert that at least one processor exists
    //assert ( procquery.random() != Realm::Processor::NO_PROC); //another possible way to do this

    for(auto it = locprocquery.begin(); it != locprocquery.end(); it++)
      _globalCTX->procs.push_back(*it);

    _globalCTX->procgroup = Realm::ProcessorGroup::create_group(_globalCTX->procs);

    (_globalCTX->cur_task).store(Realm::Processor::TASK_ID_FIRST_AVAILABLE);

    initCalled = true;
    return 0;
  }

  size_t realmGetNumProcs() {
    //DEBUG std::cout << "init runtime within realmGetNumProcs" << std::endl;
    char *dummy = "";
    realmInitRuntime(0,&dummy);

    //DEBUG std::cout << "realmGetNumProcs" << std::endl;
    if ( _globalCTX)
      return _globalCTX->numprocs;
    else
      return 0;
  }
    
  void realmArrive(Realm::Barrier& b){
    b.arrive(); 
  }

  void realmSpawn(Realm::Barrier& b,
      Realm::Processor::TaskFuncPtr func,
		  const void* args,
		  size_t arglen
    ){
    context *ctx = getRealmCTX();
    Realm::Processor::TaskFuncID taskID = ctx->cur_task++;
    Realm::Processor p = ctx->procgroup; //spawn on the group to enable Realm's magic load-balancing
    Realm::CodeDescriptor cd = Realm::CodeDescriptor(func);
    const Realm::ProfilingRequestSet prs;  //We don't care what it is for now, the default is fine
    Realm::Event e1 = p.register_task(taskID, cd, prs);
    b.alter_arrival_count(1);
    Realm::Event e2 = p.spawn(taskID, args, arglen, e1); //predicated on the completion of the task's registration
    b.arrive(1, e2); 
    return;
  }

  void realmSync(Realm::Barrier& b) {
    b.arrive(1); 
    if(Realm::Thread::self())
      b.wait(); 
    else 
      b.external_wait(); 
  }

  void realmFinalize() {
    if ( _globalCTX) {
      delete _globalCTX;
      return;
    }
    else
      return;
  }
}
