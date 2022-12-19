#!/bin/bash

source /nlu/users/jeanphilippe_corbeil/venvs/ve_tf231_ngc/bin/activate

export TRANSFORMERS_CACHE=/nlu/users/jeanphilippe_corbeil/hf_models
export HF_DATASETS_CACHE=/nlu/users/jeanphilippe_corbeil/hf_datasets
export HF_HOME=/nlu/users/jeanphilippe_corbeil/hf_hub
export RUNPATH=/nlu/users/jeanphilippe_corbeil/data_diet

# Absolute path to locate HF data diet git
export HF_NLU_DATA_DIET=$RUNPATH/hf_nlu_data_diet

cd $RUNPATH

for prune_mode in "loss" "el2n" "random"
do
for frequency in $(seq 1 5)
do
for prune_epoch in $(seq 1 6)
do
for prune_size in $(seq 0.1 0.1 0.9)
do

run_folder=all_dynamic_runs/runs_$prune_mode\_$prune_epoch\_$prune_size\_$frequency
mkdir -p $RUNPATH/$run_folder

for dataset in "snips" "atis" "slurp" "mtop"
do
mkdir -p $RUNPATH/$run_folder/$dataset
mkdir -p $RUNPATH/$run_folder/$dataset/logs
/tools/res/tools/devtools/cli/bin/jam run \
    --container research_tools/ngc-tf-2.3.1-py3:latest \
    -n "run_${dataset}_${prune_mode}_${prune_epoch}_${prune_size}_${frequency}" \
    -c "./av_run.sh" \
    --gpu_type 'v100' \
    --gpu_number 1 \
    -v "HF_NLU_DATA_DIET=$HF_NLU_DATA_DIET" \
    -v "RUNPATH=$RUNPATH" \
    -v "RUNFOLDER=$run_folder" \
    -v "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" \
    -v "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" \
    -v "HF_HOME=$HF_HOME" \
    -v "DATASET=$dataset" \
    -v "FREQUENCY=$frequency" \
    -v "PRUNE_EPOCH=$prune_epoch" \
    -v "PRUNE_SIZE=$prune_size" \
    -v "PRUNE_OFFSET=0.00" \
    -v "PRUNE_MODE=$prune_mode" \
    -o $RUNPATH/$run_folder/$dataset/logs/bft.log \
    -e $RUNPATH/$run_folder/$dataset/logs/bft.log \
    -t 1-5:1
done

done
done
done
done
