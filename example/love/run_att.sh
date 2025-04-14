#!/bin/bash
set -e 

# clean directory
rm -rf out 
mkdir -p out 

sourcedir=`pwd`/../../
echo $sourcedir

# run sem
KERNEL_TYPE=1
time $sourcedir/bin/surflove model.txt 0.01 1 500 $KERNEL_TYPE
#time $sourcedir/bin/surflove model.txt 0.01 1 2

# convert database to h5file 
python $sourcedir/scripts/binary2h5.py out/database.bin out/swd.txt out/kernels.h5

# plot  displ 
python benchmark.py 19 $KERNEL_TYPE
 
# # plot_kernel.py file period_id mode
# python plot_kernels.py out/kernels.h5 119 0

# \rm -f  *.so bench_cps.py
