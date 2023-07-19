from copy import deepcopy
from functools import partial
import time

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import TrainerCallback

from hf_nlu_data_diet.trainer.nlu_modelling import MASK_VALUE, nlu_eval_step


def compute_grad(output, parameters, loss_attr: str = "loss"):
    grads = torch.autograd.grad(getattr(output, loss_attr), parameters)
    grads = torch.concat([torch.reshape(g.detach().cpu(), (-1,)) for g in grads])
    return torch.norm(grads)


def compute_sample_grads(inputs, model, device):
    """ manually process each sample with per sample gradient """
    batch_size = inputs["input_ids"].shape[0]
    loss_attrs = ("loss",)
    sample_grads = []
    for i in range(batch_size):
        sample_grad = []
        batch_input = {k: inputs[k][i].unsqueeze(0).to(device) for k in inputs}
        output = model(**batch_input)
        for l in loss_attrs:
            sample_grad.append(compute_grad(output, model.parameters(), l))
        del batch_input
        sample_grads.append(sample_grad)
    return sample_grads


def compute_el2n(inputs, model, device, reduce_method: str, is_nlu: bool = True, all_scores: bool = False):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            output = model(**batch_input)

    logits_attr = "intent_logits" if is_nlu else "logits"
    label_key = "intent_label" if is_nlu else "labels"
    num_attr = "num_intent" if is_nlu else "num_labels"

    num_classes = getattr(model, num_attr)
    if num_classes > 1:
        p = nn.functional.softmax(getattr(output, logits_attr), dim=1)
        y = nn.functional.one_hot(batch_input[label_key], num_classes=num_classes)
    else:
        p = nn.functional.sigmoid(getattr(output, logits_attr))
        y = batch_input[label_key].unsqueeze(dim=1)
    err = p - y
    scores = torch.norm(err, dim=1)

    if not is_nlu:
        return scores.detach().cpu().numpy()

    intent_scores = scores

    p_slot = nn.functional.softmax(output.slot_logits, dim=2)
    slot_label = batch_input["slot_label"]
    slot_masks = torch.ne(slot_label, MASK_VALUE).int() * batch_input["attention_mask"]
    slot_label[(1 - slot_masks).bool()] = 0
    y_slot = nn.functional.one_hot(slot_label, num_classes=model.num_slots)
    err_slot = p_slot - y_slot
    mask_view = (slot_masks.size(dim=0), slot_masks.size(dim=1), 1)
    slot_norms = torch.norm(err_slot * slot_masks.int().view(mask_view), dim=2)

    if reduce_method == "norm":
        slot_scores = torch.norm(slot_norms, dim=1)
        total_scores = torch.sqrt(torch.pow(intent_scores, 2) + torch.pow(slot_scores, 2))
    elif reduce_method in ["sum", "mean"]:
        slot_scores = torch.sum(slot_norms, dim=1)
        if reduce_method == "mean":
            slot_scores = slot_scores / torch.sum(slot_masks, dim=1)
        total_scores = intent_scores + slot_scores
        if reduce_method == "mean":
            total_scores = total_scores / 2

    if not all_scores:
        return total_scores.detach().cpu().numpy()
    else:
        output = (total_scores.detach().cpu().numpy(), intent_scores.detach().cpu().numpy(), \
            slot_scores.detach().cpu().numpy())
        output = list(zip(*output))
        return output


def compute_persample_loss(inputs, model, device, is_nlu: bool = True):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            output = model(**batch_input)

    logits_attr = "intent_logits" if is_nlu else "logits"
    label_key = "intent_label" if is_nlu else "labels"
    num_attr = "num_intent" if is_nlu else "num_labels"

    logits = getattr(output, logits_attr)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    label = nn.functional.one_hot(batch_input[label_key], num_classes=getattr(model, num_attr))
    loss = loss_fct(logits.float(), label.float())

    if not is_nlu:
        return loss.detach().cpu().numpy()

    intent_loss = loss

    slot_logits = output.slot_logits
    slot_label = batch_input["slot_label"]
    slot_loss_fct = nn.CrossEntropyLoss(reduction="none")
    slot_masks = torch.ne(slot_label, MASK_VALUE).int() * batch_input["attention_mask"]
    slot_label[(1 - slot_masks).bool()] = 0
    slot_label = nn.functional.one_hot(slot_label, num_classes=model.num_slots)
    slot_loss = slot_loss_fct(torch.transpose(slot_logits, 1, 2).float(), torch.transpose(slot_label, 1, 2).float())

    persample_loss = model.alpha * intent_loss + (1 - model.alpha) * torch.sum(slot_masks * slot_loss, dim=1)

    return persample_loss


class ScoreCallback(TrainerCallback):
    def __init__(self, method: str, dataset, collate_fn, reduce_method: str = "norm", task: str = "nlu", all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert reduce_method in ["sum", "mean", "norm"], "'reduce_method' must 'sum', 'mean' or 'norm'."
        self.reduce_method = reduce_method
        self.method = method
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.task = task
        self.all_scores = all_scores
        self._set_compute_fn()

    def _set_compute_fn(self):
        if self.method == "loss":
            self.compute_fn = partial(compute_persample_loss, is_nlu=self.task=="nlu")
        elif self.method == "el2n":
            self.compute_fn = partial(compute_el2n, reduce_method=self.reduce_method, is_nlu=self.task=="nlu", all_scores=self.all_scores)
        elif self.method == "grand":
            self.compute_fn = compute_sample_grads

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"SCORE EVAL: {self.method}")
        model.to(args.device)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.collate_fn
        )

        start_time = time.time()

        all_grads = []
        for batch in tqdm(dataloader):
            batch_grads = self.compute_fn(inputs=batch, model=model, device=args.device)
            for i, n in zip(batch["id"], batch_grads):
                if isinstance(n, tuple):
                    elem = (int(i), *tuple(map(float, n)))
                else:
                    elem = (int(i), float(n))
                all_grads.append(elem)

        score_time = time.time() - start_time
        with open(f"{args.output_dir}/time.txt", "a") as fp:
            fp.write("%f\t%f\tscore\n" % (state.global_step, score_time))

        header = "Id\tScore"
        if self.all_scores:
            header += "\tIntent\tSlot"
        header += "\n"

        with open(f"{args.output_dir}/{self.method}_{state.epoch}.tsv", "w") as fp:
            fp.write(header)
            fp.write("\n".join("\t".join(map(str, g)) for g in all_grads))


class LossCallback(TrainerCallback):
    """
    General total loss compute per eval.
    """
    def on_evaluate(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        print("COMPUTING TRAIN LOSS")
        model.to(args.device)
        all_loss = []
        count = 0
        for inputs in tqdm(train_dataloader):
            batch_input = {k: inputs[k].to(args.device) for k in inputs}
            with torch.no_grad():
                output = model(**batch_input)
            bs = int(batch_input["input_ids"].size(dim=0))
            all_loss.append(float(output.loss.detach().cpu()) * bs)
            count += bs

        with open(f"{args.output_dir}/loss.tsv", "a") as fp:
            fp.write("%f\t%f\n" % (state.epoch, sum(all_loss)/count))


class TimeCallback(TrainerCallback):
    """
    Compute time per step (along process logging in time.txt)
    and summing at the end of the training.
    """
    def __init__(self):
        self.start_time = 0
        self.total_time = 0
    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.start_time
        with open(f"{args.output_dir}/time.txt", "a") as fp:
            fp.write("%f\t%f\tstep\n" % (state.global_step, step_time))
    def on_train_end(self, args, state, control, **kwargs):
        with open(f"{args.output_dir}/time.txt", "r") as fp:
            times = [tuple(f.strip().split("\t")) for f in fp.readlines()]
        self.total_time = sum([float(t) for _, t, _ in times])


class ForgetScoreCallback(TrainerCallback):
    """
    Compute forget score at the end based
    on full label match at each step.
    """
    def __init__(self, output_dir: str, train_dataset, collate_fn, task: str = "nlu"):
        self.forget_pattern = [1, 0] # Forget event is going from learned to unlearned.
        self.train_dataset = train_dataset
        self.train_len = len(train_dataset)
        self.collate_fn = collate_fn
        self.output_dir = output_dir
        self.task = task

        # Create ouput file.
        self.file_path = f"{output_dir}/forget_event.tsv"
        with open(self.file_path, "w") as fp:
            fp.write("Id\tStep\tEpoch\tEvent\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        bs = args.per_device_train_batch_size
        steps = int(self.train_len / bs) + 1
        current_step = state.global_step % steps
        batch = self.train_dataset.select(range(bs*current_step, min(bs*(current_step + 1), self.train_len)))
        batch = self.collate_fn(list(batch))

        model.to(args.device)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        full_match = nlu_eval_step(model, batch, fullseq_only=True)

        with open(self.file_path, "a") as fp:
            for i, j in zip(batch["id"].detach().cpu().tolist(), full_match):
                fp.write("%d\t%d\t%d\t%d\n" % (i, current_step, state.epoch, j))

    def _compute_forget_scores(self, df):
        pattern = self.forget_pattern
        n = len(pattern)
        scores = df.rolling(window=n, min_periods=n).apply(lambda x: (x==pattern).all()).fillna(0).sum()
        return scores

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.read_csv(self.file_path, sep="\t")
        df = df.drop(["Step"], axis=1)
        event_df = pd.pivot_table(df, index="Epoch", columns="Id", values="Event")
        forget_scores = self._compute_forget_scores(event_df)
        forget_scores = forget_scores.reset_index().rename({0: "Score"}, axis=1)
        forget_scores.to_csv(f"{self.output_dir}/forget_scores.tsv", sep="\t")
