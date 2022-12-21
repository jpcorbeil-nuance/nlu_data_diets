#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from nuance_tensorflow_utils.executors import RunAsCUDA

from hf_nlu.trainer.trainer_manager import TrainerManager
from hf_nlu.trainer.prune_utils import PruneConfig
from hf_nlu.trainer.nlu_modelling import RobertaForNLU, RobertaNLUConfig, MASK_VALUE
from hf_nlu.trainer.dataset_utils import format_dataset
from hf_nlu.trainer.utils import rev_dict


def parse_args(raw_args=None):
    # FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/atis/nlu_data", help="Path in which to find the dataset folders.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name or path of the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path directory.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Seed integer for pseudo-random numbers.")
    parser.add_argument("--eval_per_epoch", type=int, default=1, help="Number of evaluations per epoch.")
    parser.add_argument("--frequency", type=int, default=1, help="Swap data pruninng every N epochs.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs in total.")
    parser.add_argument("--prune_epoch", type=float, default=40, help="Number of epochs before pruning.")
    parser.add_argument("--prune_mode", type=str, default="el2n", help="Mode of pruning.")
    parser.add_argument("--prune_size", type=float, default=0.5, help="Proportion of samples to prune.")
    parser.add_argument("--prune_offset", type=float, default=0.0, help="Offset to proportion of samples to prune.")
    parser.add_argument("--prune_avg_mode", type=str, default="ema", help="Mode of average scores (ema, avg, none).")
    parser.add_argument("--prune_avg_window_size", type=int, default=1, help="Number of reported scores before 'prune_epoch' to average on (only 'prune_avg_mode' == 'avg').")
    parser.add_argument("--prune_ema_alpha", type=float, default=0.8, help="Exponential Moving Average coefficient on prune scores.")
    parser.add_argument("--prune_ema_var_coef", type=float, default=1.0, help="Coefficient on variance for prune scores.")
    parser.add_argument("--prune_ema_use_std", action='store_false', help="Use std instead of variance (default, std).")
    parser.add_argument("--bs", type=int, default=32, help="Batch size to use in training.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Amount of samples to provide in one batch.")
    return parser.parse_args(raw_args)


def tokenize_and_align_labels(examples, tokenizer, slot2id: dict, intent2id: dict):
    """
    Tokenize text and align whitespace delimited slots into subwords with masked
    token for special tokens and subsequent subwords.
    """
    global MASK_VALUE
    tokenized_inputs = tokenizer(examples["text"], is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["slots"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to MASK_VALUE.
            if word_idx is None:
                label_ids.append(MASK_VALUE)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                current_label = label[word_idx]
                label_ids.append(slot2id.get(current_label, MASK_VALUE))
            else:
                label_ids.append(MASK_VALUE)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["intent_label"] = list(map(lambda x: intent2id.get(x, -100), examples["intent"]))
    tokenized_inputs["slot_label"] = labels
    return tokenized_inputs


def generate_dataset(dataset_path: str, tokenizer, random_seed: int = 1234):
    """
    Generate HF datasets from parquet dataframes train/test.
    """
    dataset = load_dataset("parquet", data_files={
        'train': f'{dataset_path}/train.parquet',
        'test': f'{dataset_path}/test.parquet'
    })
    dataset.shuffle(random_seed)

    slot_list = ["O"] + list(set([s for S in dataset["train"]["slots"] for s in S.split() if s != "O"]))
    slot2id = {s: i for i, s in enumerate(slot_list)}
    intent_list = list(set(dataset["train"]["intent"]))
    intent2id = {I: i for i, I in enumerate(intent_list)}

    dataset = dataset.map(lambda x: {'text': x['text'].split()})
    dataset = dataset.map(lambda x: {'slots': x['slots'].split()})
    dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, slot2id, intent2id), batched=True)
    dataset = format_dataset(dataset)
    return dataset["train"], dataset["test"], rev_dict(intent2id), rev_dict(slot2id)


def main(raw_args=None):
    args_in = parse_args(raw_args=raw_args)

    # Lock GPU if there is.
    temp_model = RobertaModel(config=RobertaConfig())
    temp_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Assign args to variables.
    model_name = args_in.model_name
    dataset_path = args_in.dataset_path
    random_seed = args_in.random_seed
    output_dir = args_in.output_path
    eval_per_epoch = args_in.eval_per_epoch
    frequency = args_in.frequency
    learning_rate = args_in.lr
    bs = args_in.bs

    epochs = args_in.epochs
    prune_epoch = args_in.prune_epoch
    final_n_epochs = epochs - prune_epoch
    assert final_n_epochs >= 0, "'epochs' must be greater or equal to 'prune_epoch'."

    prune_avg_mode = args_in.prune_avg_mode
    prune_window_size = args_in.prune_avg_window_size
    assert prune_window_size > 0 and prune_window_size <= eval_per_epoch * prune_epoch, "prune_windows_size muste be between [1, 'eval_per_epoch' * 'prune_epoch']"
    prune_size = args_in.prune_size
    assert prune_size >= 0.0 and prune_size <= 1.0, "'prune_size' must be in range: [0.0, 1.0]."
    prune_offset = args_in.prune_offset
    assert prune_offset >= 0.0 and prune_offset <= 1.0, "'prune_offset' must be in range: [0.0, 1.0]."
    assert 1.0 - prune_offset >= prune_size, "'prune_size' must be lower or equal to 1.0 - 'prune_offset'"
    prune_mode = args_in.prune_mode
    assert prune_mode in ["grand", "el2n", "loss", "random"], "'prune_mode' must be: grand, el2n, loss or random."

    ema_alpha=args_in.prune_ema_alpha
    ema_var_coef=args_in.prune_ema_var_coef
    ema_use_std=args_in.prune_ema_use_std

    prune_config = PruneConfig(
        mode=prune_mode,
        size=prune_size,
        epoch=prune_epoch,
        offset=prune_offset,
        frequency=frequency,
        avg_mode=prune_avg_mode,
        avg_window_size=prune_window_size,
        ema_alpha=ema_alpha,
        ema_var_coef=ema_var_coef,
        ema_use_std=ema_use_std
    )

    print("LOAD TOKENIZER AND DATASET")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    train, test, id2intent, id2slots = generate_dataset(dataset_path, tokenizer, random_seed)

    # Set trainer args.
    args = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=prune_epoch,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=2 * bs,
        learning_rate=learning_rate,
        seed=random_seed,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        disable_tqdm=False,
        report_to=None,
        prediction_loss_only=True,
        remove_unused_columns=False
    )

    print("LOAD MODEL AND CONFIG")
    config = RobertaNLUConfig.from_pretrained(model_name)
    config.update(dict(
            n_reinit_last_layer=1,
            num_intent=len(id2intent),
            id2intent=id2intent,
            num_slots=len(id2slots),
            id2slots=id2slots
    ))
    model = RobertaForNLU.from_pretrained(model_name, config=config)

    train_manager = TrainerManager(
        args = args,
        epochs = epochs,
        eval_per_epoch = eval_per_epoch,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train,
        test_dataset = test,
        prune_config=prune_config
    )

    # Unlock GPU if it is
    temp_model.to("cpu")
    del temp_model

    train_manager.run()


@RunAsCUDA(n_devices=1)
def executor():
    main()


if __name__ == "__main__":
    executor()
