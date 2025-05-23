#!/bin/bash
set -e 

# clean directory
rm -rf out 
mkdir -p out 

sourcedir=`pwd`/../../
echo $sourcedir


# run sem
time $sourcedir/bin/surfvti_ac model.txt  0.01 5 100

# convert database to h5file #
#python $sourcedir/scripts/binary2h5.py out/database.bin out/swd.txt out/kernels.h5

# run benckmark
\rm -f  *.so bench_cps.py 
ln -s $sourcedir/lib/cps* . 
\cp $sourcedir/scripts/bench_cps.py .

python bench_cps.py 2

# plot  displ 
python $sourcedir/scripts/plot_disp.py 5

# # plot_kernel.py file period_id mode
# python plot_kernels.py out/kernels.h5 20 0

\rm -f  *.so bench_cps.py
