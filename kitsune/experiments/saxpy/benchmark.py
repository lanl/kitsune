###########
#
#  NOTE: You will need at least Python 3.7 for this script.
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

# Array of problem sizes (trip counts) to benchmark.
array_sizes = []
for x in (2**p for p in range(24, 30)):
  array_sizes.append(x)

executables = ["saxpy-forall.cuda."+march,
               "saxpy-kokkos."+march]
csv_filename = str("saxpy-benchmark-") + march + "-" + date_str + ".csv";

header = ['Size', 'Benchmark', 'Time']

with open(csv_filename, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(header)
  row=[]

  for num in array_sizes:
    print("problem size: ", "{:,}".format(num))
    for exe in executables:
      row.append(num)
      row.append(exe)
      print("  ", exe)
      arg0 = "./" + exe
      kernel_runtime = 0.0
      overall_runtime = 0.0
      print("    ", end="")
      for rc in range(NUM_RUNS):
        result = subprocess.run([arg0, str(num)], capture_output=True, text=True)
        output = result.stdout.split()
        kernel_runtime = kernel_runtime + float(output[5])
        overall_runtime = overall_runtime + float(output[8])
        print("#", end="",flush=True)
      print("")
      kernel_runtime = kernel_runtime / NUM_RUNS
      overall_runtime = overall_runtime / NUM_RUNS
      print("    average kernel runtime:", float("{:.6f}".format( kernel_runtime)))
      print("    average overall runtime:", float("{:.6f}".format(overall_runtime)), flush=True)
      row.append(kernel_runtime)
      writer.writerow(row)
      row.clear()
    print("")

print("benchmark results saved to: ", csv_filename)

df = pd.read_csv(csv_filename)
plotdf = df.pivot(index='Size', columns='Benchmark', values='Time')
plotdf.plot(kind='bar', figsize=(15, 13))
plt.xlabel('Array Size (# of elements)', fontsize=14, )
plt.ylabel('Total Kernel Execution Times (secs)', fontsize=14)
plt.title('Saxpy Benchmark', fontsize=20, fontweight='bold')
pdf_name = str("plots/saxpy-benchmark-") + march + "-" + date_str + ".pdf"
plt.savefig(pdf_name)
jpg_name = str("plots/saxpy-benchmark-") + march + "-" + date_str + ".jpg"
plt.savefig(jpg_name)
