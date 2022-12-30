#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$HF_NLU_DATA_DIET

random_seed=$RANDOM;
mkdir -p $RUNFOLDER/$DATASET
mkdir -p $RUNFOLDER/$DATASET/$random_seed
echo "New RUN - $random_seed";

python -m hf_nlu.nlu-training \
--output_path $RUNFOLDER/$DATASET/$random_seed \
--dataset_path $RUNPATH/datasets/$DATASET/nlu_data \
--static_score_file_path $RUNPATH/scores/$SCORE_PATH \
--random_seed $random_seed \
--frequency $FREQUENCY \
--prune_epoch $PRUNE_EPOCH \
--prune_size $PRUNE_SIZE \
--prune_offset $PRUNE_OFFSET \
--prune_mode $PRUNE_MODE \
--prune_ema_var_coef $VAR_COEF
# --prune_ema_use_var


# Remove heavy checkpoints
rm -r $RUNFOLDER/$DATASET/$random_seed/checkpoint-*/
