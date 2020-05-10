#!/usr/bin/env bash

experiments=${1}
model_name=${2}
makefile=${3}

make -f ${makefile} run source_dataset='M4' \
     source_subset='yearly' \
     target_dataset='M4' \
     target_subset='yearly' \
     frequency='12M' \
     horizon=6 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='quarterly' \
     target_dataset='M4' \
     target_subset='quarterly' \
     frequency='3M' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='monthly' \
     target_dataset='M4' \
     target_subset='monthly' \
     frequency='M' \
     horizon=18 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='weekly' \
     target_dataset='M4' \
     target_subset='weekly' \
     frequency='7D' \
     horizon=13 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='daily' \
     target_dataset='M4' \
     target_subset='daily' \
     frequency='D' \
     horizon=14 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='M4' \
     target_subset='hourly' \
     frequency='H' \
     horizon=48 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M3' \
     source_subset='yearly' \
     target_dataset='M3' \
     target_subset='yearly' \
     frequency='12M' \
     horizon=6 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M3' \
     source_subset='quarterly' \
     target_dataset='M3' \
     target_subset='quarterly' \
     frequency='3M' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M3' \
     source_subset='monthly' \
     target_dataset='M3' \
     target_subset='monthly' \
     frequency='M' \
     horizon=18 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M3' \
     source_subset='others' \
     target_dataset='M3' \
     target_subset='others' \
     frequency='H' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Tourism' \
     source_subset='yearly' \
     target_dataset='Tourism' \
     target_subset='yearly' \
     frequency='12M' \
     horizon=4 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Tourism' \
     source_subset='quarterly' \
     target_dataset='Tourism' \
     target_subset='quarterly' \
     frequency='3M' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Tourism' \
     source_subset='monthly' \
     target_dataset='Tourism' \
     target_subset='monthly' \
     frequency='M' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Electricity' \
     source_subset='"2014-01-01 00:2014-03-31 00"' \
     target_dataset='Electricity' \
     target_subset='"2014-01-01 00:2014-03-31 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Electricity' \
     source_subset='"2014-01-01 00:2014-09-01 00"' \
     target_dataset='Electricity' \
     target_subset='"2014-01-01 00:2014-09-01 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Electricity' \
     source_subset='"2014-03-25 00:2014-12-25 01"' \
     target_dataset='Electricity' \
     target_subset='"2014-03-25 00:2014-12-25 01"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Traffic' \
     source_subset='"2008-01-02 01:2008-01-14 00"' \
     target_dataset='Traffic' \
     target_subset='"2008-01-02 01:2008-01-14 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Traffic' \
     source_subset='"2008-01-02 01:2008-06-15 00"' \
     target_dataset='Traffic' \
     target_subset='"2008-01-02 01:2008-06-15 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='Traffic' \
     source_subset='"2008-09-02 01:2009-03-24 01"' \
     target_dataset='Traffic' \
     target_subset='"2008-09-02 01:2009-03-24 01"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

##### Transfer Learning ######
make -f ${makefile} run source_dataset='M4' \
     source_subset='yearly' \
     target_dataset='M3' \
     target_subset='yearly' \
     frequency='12M' \
     horizon=6 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='yearly' \
     target_dataset='Tourism' \
     target_subset='yearly' \
     frequency='12M' \
     horizon=4 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='quarterly' \
     target_dataset='M3' \
     target_subset='quarterly' \
     frequency='3M' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='quarterly' \
     target_dataset='Tourism' \
     target_subset='quarterly' \
     frequency='3M' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='quarterly' \
     target_dataset='M3' \
     target_subset='others' \
     frequency='H' \
     horizon=8 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='monthly' \
     target_dataset='M3' \
     target_subset='monthly' \
     frequency='M' \
     horizon=18 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'


make -f ${makefile} run source_dataset='M4' \
     source_subset='monthly' \
     target_dataset='Tourism' \
     target_subset='monthly' \
     frequency='M' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Electricity' \
     target_subset='"2014-01-01 00:2014-03-31 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Electricity' \
     target_subset='"2014-01-01 00:2014-09-01 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Electricity' \
     target_subset='"2014-03-25 00:2014-12-25 01"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Traffic' \
     target_subset='"2008-01-02 01:2008-01-14 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Traffic' \
     target_subset='"2008-01-02 01:2008-06-15 00"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'

make -f ${makefile} run source_dataset='M4' \
     source_subset='hourly' \
     target_dataset='Traffic' \
     target_subset='"2008-09-02 01:2009-03-24 01"' \
     frequency='H' \
     horizon=24 \
     model_name="${model_name}" \
     experiments="${experiments}" \
     storage_path='/mnt/experiment'