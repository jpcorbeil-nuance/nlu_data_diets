import os
from copy import deepcopy

from transformers import Trainer, TrainingArguments, AdamW, get_constant_schedule
from transformers import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import TensorBoardCallback

from hf_nlu_data_diet.trainer.nlu_modelling import DataCollatorForNLU, nlu_evaluate
from hf_nlu_data_diet.trainer.callbacks import ScoreCallback, LossCallback, TimeCallback, ForgetScoreCallback
from hf_nlu_data_diet.trainer.prune_utils import PruneConfig, PruneScoreManager
from hf_nlu_data_diet.trainer.dataset_utils import format_nlu_dataset, format_classification_dataset
from hf_nlu_data_diet.trainer.utils import save_evaluation


def compute_steps(n_train_dataset: int, batch_size: int):
    return int(n_train_dataset / batch_size) + 1


class TrainerManager:
    """
    Handle HF Trainer for periodic restart of training with variable trainset.
    """
    def __init__(self, args: dict, epochs: int,
            model, tokenizer, train_dataset, test_dataset, prune_config: PruneConfig,
            eval_per_epoch: int = 1, track_loss: bool = False, data_collator=None,
            compute_metrics=None, max_seq_length: int = 50, task: str = "nlu",
            all_scores: bool = False):
        self.args = args
        self.epochs = epochs
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_per_epoch = eval_per_epoch
        self.prune_manager = PruneScoreManager(output_dir=args.get("output_dir"), config=prune_config)
        self.track_loss = track_loss
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.max_seq_length = max_seq_length
        self.task = task
        self.all_scores = all_scores
        self.format_dataset = format_nlu_dataset if task == "nlu" else format_classification_dataset

        self._init_trainer()

    def _init_opt(self):
        """Initialize optimizer and schedule."""
        adam = AdamW(self.model.parameters(), lr=self.args["learning_rate"])
        schedule = get_constant_schedule(adam)
        return adam, schedule

    def _compute_eval_steps(self, use_prune_size: bool = False):
        """Compute evaluation steps based on trainset length and eval_per_epoch."""
        if use_prune_size:
            n_train_dataset = int(self.prune_manager.config.prune_size * len(self.train_dataset))
        else:
            n_train_dataset = len(self.train_dataset)
        nb_steps = compute_steps(n_train_dataset, self.args["per_device_train_batch_size"])
        return round(nb_steps / self.eval_per_epoch)

    def _generate_training_args(self) -> TrainingArguments:
        """Generate training arguments for trainer based on args."""
        eval_steps = self._compute_eval_steps()
        return TrainingArguments(
            warmup_ratio=0.0,
            evaluation_strategy="no",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            logging_strategy="steps",
            logging_steps=eval_steps,
            **self.args
        )

    def _sort_trainset_by_length(self):
        trainset = deepcopy(self.train_dataset)
        trainset.reset_format()
        trainset = trainset.map(lambda x: {"length": len(x["input_ids"])})
        trainset = trainset.sort("length")
        return self.format_dataset(trainset)

    def _init_trainer(self):
        """Initialize trainer with callbacks."""
        train_args = self._generate_training_args()
        opt_tup = self._init_opt()

        if self.data_collator is None:
            collate_fn = DataCollatorForNLU(
                tokenizer=self.tokenizer,
                padding='longest',
                max_length=self.max_seq_length
            )
        else:
            collate_fn = self.data_collator

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=train_args,
            train_dataset=deepcopy(self.train_dataset),
            eval_dataset=deepcopy(self.test_dataset),
            optimizers=opt_tup,
            data_collator=collate_fn,
            compute_metrics=self.compute_metrics
        )

        self.trainer.callback_handler.remove_callback(TensorBoardCallback)
        if self.track_loss:
            self.trainer.add_callback(LossCallback)

        sort_trainset = self._sort_trainset_by_length()

        prune_mode = self.prune_manager.config.prune_mode
        if prune_mode in ["grand", "el2n", "loss"]:
            # Reduce padding and runtime in forward pass by sorting.
            score_callback = ScoreCallback(
                method=prune_mode,
                dataset=sort_trainset,
                collate_fn=collate_fn,
                task=self.task,
                all_scores=self.all_scores
            )
            self.trainer.add_callback(score_callback)
        elif prune_mode == "forget":
            score_callback = ForgetScoreCallback(
                output_dir=self.args.get("output_dir"),
                train_dataset=deepcopy(self.train_dataset),
                collate_fn=collate_fn,
                task=self.task
            )
            self.trainer.add_callback(score_callback)

        self.trainer.add_callback(TimeCallback)

    def _update_traindata(self):
        """Fetch scores and filter temporary trainset to overwrite in trainer."""
        train_dataset = deepcopy(self.train_dataset)
        train_dataset.reset_format()
        score_dict, bound_tup = self.prune_manager.get_scores_and_bounds(sample_ids=train_dataset["id"])
        train_dataset = train_dataset.map(lambda x: {"Score": score_dict[x["id"]]})
        train_filtered = train_dataset.filter(lambda x: x["Score"] >= bound_tup[0] and x["Score"] <= bound_tup[1])
        train_filtered = self.format_dataset(train_filtered)
        self.trainer.train_dataset = deepcopy(train_filtered)

    def _compensate_trainer_state(self):
        """Change trainer state to account for change in train size."""
        last_ckpt = get_last_checkpoint(self.trainer.args.output_dir)
        trainer_state_path = os.path.join(last_ckpt, "trainer_state.json")
        state = TrainerState.load_from_json(trainer_state_path)
        state.global_step = int(self.prune_manager.config.prune_size * state.global_step)
        state.save_to_json(trainer_state_path)

    def _adjust_eval_steps(self):
        """Re-adjust eval steps based on new trainset length."""
        new_eval_steps = self._compute_eval_steps(use_prune_size=True)
        self.trainer.args.eval_steps = new_eval_steps
        self.trainer.args.save_steps = new_eval_steps
        self.trainer.args.logging_steps = new_eval_steps

    def _prune_run(self, update_n_epochs: int, not_first: bool = True):
        """One run of pruned data."""
        print("PRUNING TRAINING DATA...")
        self._update_traindata()

        # Update epoch state
        self.trainer.args.num_train_epochs += update_n_epochs

        print("RESTARTING TRAINING...")
        self.trainer.train(resume_from_checkpoint=not_first)

    def run(self):
        """Full run of dynamical pruning"""
        epochs = self.epochs
        prune_epoch = self.prune_manager.config.prune_epoch

        print("TRAINING...")
        if prune_epoch > 0:
            self.trainer.train()

        if epochs > prune_epoch:
            # Just need to compensate and adjust once.
            if prune_epoch > 0:
                self._compensate_trainer_state()
            elif prune_epoch == 0:
                old_prune_mode = self.prune_manager.config.prune_mode
                self.prune_manager.config.prune_mode = "random"

            self._adjust_eval_steps()

            frequency = self.prune_manager.config.prune_frequency
            rest_epochs = epochs - prune_epoch
            if frequency < rest_epochs:
                num_prune = int(rest_epochs / frequency)
                for p in range(num_prune):
                    self._prune_run(frequency, not_first=not (prune_epoch == 0 and p == 0))
                    if prune_epoch == 0 and self.prune_manager.config.prune_mode == "random":
                        self.prune_manager.config.prune_mode = old_prune_mode

            # If some epochs remain before full training ( < frequency).
            if epochs > self.trainer.args.num_train_epochs:
                self._prune_run(epochs-self.trainer.args.num_train_epochs)

        # Fetch total time from TimeCallback
        total_sec = self.trainer.callback_handler.callbacks[-1].total_time

        output_dir = self.args.get("output_dir")
        final_eval = {"runtime": total_sec, "epochs": epochs, "learning_rate": self.args["learning_rate"], "fp16": self.args["fp16"]}
        if self.task == "nlu":
            print("EVALUATING...")
            eval_metrics = nlu_evaluate(self.model, self.trainer.get_eval_dataloader(), self.trainer.args.device)
            final_eval.update(eval_metrics)
        save_evaluation(final_eval, output_dir)
        self.prune_manager.config.save_to_json(output_dir)

        print("Done!")
