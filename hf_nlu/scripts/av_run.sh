#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$HF_NLU_DATA_DIET

# nvidia-smi;
# CUDA_VISIBLE_DEVICES=$(python -m hf_nlu.extract_gpu)
# echo $CUDA_VISIBLE_DEVICES

random_seed=$RANDOM;
mkdir -p $RUNFOLDER/$DATASET
mkdir -p $RUNFOLDER/$DATASET/$random_seed
echo "New RUN - $random_seed";

# debug with CUDA_LAUNCH_BLOCKING=1
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python -m hf_nlu.nlu-training \
--output_path $RUNFOLDER/$DATASET/$random_seed \
--dataset_path $RUNPATH/datasets/$DATASET/nlu_data \
--random_seed $random_seed \
--frequency $FREQUENCY \
--prune_epoch $PRUNE_EPOCH \
--prune_size $PRUNE_SIZE \
--prune_offset $PRUNE_OFFSET \
--prune_mode $PRUNE_MODE

# Remove heavy checkpoints
rm -r $RUNFOLDER/$DATASET/$random_seed/checkpoint-*/
