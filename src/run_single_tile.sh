#!/bin/bash
 
#PBS -P u46
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=496gb
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/os22+gdata/xc0
 
module load python3/3.10.0
source /g/data/os22/chad_tmp/AusENDVI/env/py310/bin/activate

python3 /g/data/os22/chad_tmp/Aus_phenology/src/batch_run_phenology_analysis.py $TILENAME