#!/bin/bash

source /nlu/users/jeanphilippe_corbeil/venvs/ve_tf231_ngc/bin/activate

export TRANSFORMERS_CACHE=/nlu/users/jeanphilippe_corbeil/hf_models
export HF_DATASETS_CACHE=/nlu/users/jeanphilippe_corbeil/hf_datasets
export HF_HOME=/nlu/users/jeanphilippe_corbeil/hf_hub
export RUNPATH=/nlu/users/jeanphilippe_corbeil/data_diet

# Absolute path to locate HF data diet git
export HF_NLU_DATA_DIET=$RUNPATH/hf_nlu_data_diet

for prune_mode in "el2n"
do
for frequency in 4
do
for prune_epoch in 1
do
for prune_size in 0.5
do

run_folder=$RUNPATH/test/runs_$prune_mode\_$prune_epoch\_$prune_size\_$frequency
mkdir -p $run_folder

for dataset in "atis"
do
mkdir -p $run_folder/$dataset
mkdir -p $run_folder/$dataset/logs
/tools/res/tools/devtools/cli/bin/jam run \
    --container research_tools/ngc-tf-2.3.1-py3:latest \
    -n "run_${dataset}_${prune_mode}_${prune_epoch}_${prune_size}_${frequency}" \
    -c "./av_run.sh" \
    --gpu_type 'v100|p6000' \
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
    -v "PRUNE_OFFSET=0.0" \
    -v "PRUNE_MODE=$prune_mode" \
    -o $run_folder/$dataset/logs/bft.log \
    -e $run_folder/$dataset/logs/bft.log
    # -t 1-5:1
done

done
done
done
done
