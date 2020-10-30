#!/bin/bash
#
#
#PBS -l nodes=1:ppn=16,walltime=72:00:00



# Define working directory
export WORK_DIR=$HOME/sfc_rescaled/sfc_g_64w

# Define executable
export EXE="/usr/bin/env Rscript snep_growing.R"

# Add R module
module add languages/R-3.0.2

# Change into working directory
cd $WORK_DIR

# Do some stuff
echo JOB ID: $PBS_JOBID
echo Working Directory: `pwd`
echo Start Time: `date`

# Execute code
$EXE

echo End Time: `date`



sleep 20
