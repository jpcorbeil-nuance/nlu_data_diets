# NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks

## Execute on GLUE datasets

The script for the trainings on GLUE was derived from the HF's implementation. Please check it out to be familiar with their parameters. We added all the pruning parameters by leveraging our PruneConfig class.

    python -m hf_nlu.glue_training \
      --model_name_or_path roberta-base \
      --seed $random_seed \
      --dataset_name $DATASET  \
      --task_name $DATASET  \
      --do_train \
      --do_eval \
      --fp16 \
      --max_seq_length 256 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 10 \
      --prune_mode $PRUNE_MODE \
      --prune_epoch $PRUNE_EPOCH \
      --prune_size $PRUNE_SIZE \
      --prune_frequency $FREQUENCY \
      --output_dir $RUNFOLDER/$DATASET/$random_seed

## Execute on NLU datasets

Check CLI help for more details on each parameter.

    python -m hf_nlu.nlu_training \
      --output_path $RUNFOLDER/$DATASET/$random_seed \
      --dataset_path $RUNPATH/datasets/$DATASET/nlu_data \
      --random_seed $random_seed \
      --frequency $FREQUENCY \
      --prune_mode $PRUNE_MODE \
      --prune_epoch $PRUNE_EPOCH \
      --prune_size $PRUNE_SIZE \
      --prune_offset $PRUNE_OFFSET \
      --all_scores

## Citation

    @inproceedings{attendu-corbeil-2023-nlu,
        title = "NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks",
        author = "Attendu, Jean-michel and Corbeil, Jean-philippe",
        booktitle = "Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada (Hybrid)",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.sustainlp-1.9",
        pages = "129--146",
    }
