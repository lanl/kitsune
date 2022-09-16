#!/bin/bash

for exe in saxpy-kokkos.x86_64 saxpy-forall.cuda.x86_64
do
  echo "running profiler..."
  ncu --section SpeedOfLight \
      --section SpeedOfLight_RooflineChart \
      --section SchedulerStats \
      --section ComputeWorkloadAnalysis \
      --section WarpStateStats \
      --section LaunchStats \
      --section InstructionStats \
      --section SourceCounters \
      --section Occupancy \
      --section MemoryWorkloadAnalysis \
      --page details \
      --print-summary per-gpu \
      --details-all \
      --log-file ncu-$exe.log \
      --export $exe.%i $exe
done 

