# HF NLU DATA DIET

This library is a HuggingFace implementation of dynamic data pruning with a Pytorch backend.

Example of execution from main script *nlu-training.py*:

    python -m hf_nlu.nlu-training \
        --output_path /.../output/atis/1234 \
        --dataset_path /.../datasets/atis/nlu_data \
        --random_seed 1234


    python -m hf_nlu.nlu-training \
        --output_path /.../output/atis/1234 \
        --dataset_path /.../datasets/atis/nlu_data \
        --random_seed 1234 \
        --frequency 4 \
        --prune_epoch 4 \
        --prune_size 0.5 \
        --prune_mode el2n

Check the following script for thorough usage on UNIX: *hf_nlu/scripts/launch_jam.sh*.
