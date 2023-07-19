# NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks

## Datasets

### GLUE

We leverage the `datasets` library from HF to load and parse the data as with the original HF GLUE training script.

### Joint NLU tasks

All datasets were gathered using the following links and put inside their respective folders `hf_nlu_data_diet/datasets/$DATASET_NAME/raw/`.

  - [ATIS](https://github.com/howl-anderson/ATIS_dataset/tree/master): from ATIS_dataset repository.
  - [SNIPS](https://github.com/monologg/JointBERT): source from the JointBERT github's SNIPS dataset.
  - [MTOP](https://fb.me/mtop_dataset): just the english segment.
  - [SLURP](https://github.com/alexa/massive): we took the English part of the MASSIVE dataset (based on SLURP) from Amazon Alexa for simplicity.

Then, we used their parse script provided in each dataset folder to generate the train/test parquet files using standard NLU formatting (columns are "id" for index in dataset, "text" for plain utterance, "intent" for intent label and "slots" for whitespace-delimited label sequence). We load the parquets with the `datasets` library in the training script.

## Execute

### GLUE

The script for the trainings on GLUE was derived from the HF's implementation. Please check it out to be familiar with their parameters. We added all the pruning parameters by leveraging our PruneConfig class.

    python -m hf_nlu_data_diet.glue_training \
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

### Joint NLU

Check CLI help for more details on each parameter.

    python -m hf_nlu_data_diet.nlu_training \
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
