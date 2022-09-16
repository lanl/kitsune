###########
#
# NOTE: You will need at least Python 3.7 for this script.
#
# Run the set of benchmarks in the directory and capture the results
# in a CSV output file.  The file is then read and processed via pandas
# and matplotlib to provide a plot of the performance across all the
# benchmarks.
#
# The code to gather run time data is dependent upon the output format
# from the benchmarks.  Any changes to the output format will likely
# break this script.
#
import subprocess
import platform
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


# Grab the local machine's architecture details and the date+time to
# create a unique filename for this benchmark run.
march = platform.machine()
now = datetime.now()
date_str = now.strftime("%m-%d-%Y_%H:%M")
print(date_str)

# Number of times to run each benchmark to get an average
# execution time.
NUM_RUNS = 5

# The raytracer's runtime varies based on two input parameters.  The
# first is the number of samples per pixel and the second is the
# dimensions of the output image.  We set up a loop over varying
# ranges of values for these two parameters.

sample_counts = []
for sc in (2**p for p in range(2, 6)):
  sample_counts.append(sc)

img_width = 1920
img_height = 1080

executables = ["raytrace-cuda.clang."+march,
               "raytrace-cuda.nvcc."+march,
               "raytrace-kokkos.clang."+march,
               "raytrace-kokkos.nvcc."+march,
               "raytrace-openmp.clang."+march,
               "raytrace-forall.cuda."+march]

csv_filename = str("raytracer-benchmark-") + march + "-" + date_str + ".csv";

header = ['Samples', 'Benchmark','Time']

with open(csv_filename, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(header)
  row=[]

  for s in sample_counts:
    print("number of samples: ", s)
    for exe in executables:
      row.append(s)
      row.append(exe)
      print("  ", exe)
      arg0 = "./" + exe
      kernel_runtime = 0.0
      print("    ", end="", flush=True)
      for rc in range(NUM_RUNS):
        result = subprocess.run([arg0,str(s),str(img_width),str(img_height)],  capture_output=True, text=True)
        kernel_runtime = kernel_runtime + float(result.stdout)
        print('#', end='', flush=True)
      print("")
      kernel_runtime = kernel_runtime / float(NUM_RUNS)
      print("  kernel runtime:", kernel_runtime)
      row.append(kernel_runtime)
      writer.writerow(row)
      row.clear()
    print(" ")

print("benchmark results saved to: ", csv_filename)

df=pd.read_csv(csv_filename)
plotdf=df.pivot(index='Samples', columns='Benchmark', values='Time')
plotdf.plot(kind = 'bar', figsize = (15, 13))
plt.xlabel('Samples per Pixel', fontsize = 14, )
plt.ylabel('Kernel Execution Time (secs)', fontsize=14)
plt.title('Raytracer Benchmark', fontsize=20, fontweight='bold')
pdf_name=str("plots/raytracer-benchmark-") + march + "-" + date_str + ".pdf"
plt.savefig(pdf_name)
jpg_name=str("plots/raytracer-benchmark-") + march + "-" + date_str + ".jpg"
plt.savefig(jpg_name)
