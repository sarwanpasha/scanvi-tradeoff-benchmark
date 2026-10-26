#!/bin/bash
# End-to-end pipeline. Each line of the task files can also be run as a separate cluster job.
set -e
mkdir -p data/raw data/std emb res summary
python lupus_download.py
python hlca_download.py
for d in pbmc_fine spleen_lymph heart hlca lupus; do python prep_data.py $d; done
python make_tasks.py
while read -r args; do python train.py $args; done < tasks_train.txt
while read -r args; do python celltypist_run.py $args; done < tasks_ct.txt
while read -r args; do python evaluate.py $args; done < tasks_eval.txt
python aggregate.py > summary/RESULTS.txt
python lupus_bio.py > summary/LUPUS.txt
