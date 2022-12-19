import time

import torch
from torch import nn
from tqdm import tqdm
from transformers import TrainerCallback

from hf_nlu.trainer.nlu_modelling import MASK_VALUE


def compute_grad(output, parameters, loss_attr: str = "loss"):
    grads = torch.autograd.grad(getattr(output, loss_attr), parameters) # allow_unused=True
    grads = torch.concat([torch.reshape(g.detach().cpu(), (-1,)) for g in grads]) # if g is not None
    return torch.norm(grads)


def compute_sample_grads(inputs, model, device):
    """ manually process each sample with per sample gradient """
    batch_size = inputs["input_ids"].shape[0]
    loss_attrs = ("loss",) # "intent_loss", "slot_loss"
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


class GradientNormCallback(TrainerCallback):
    def __init__(self, dataset: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_train_end(self, args, state, control, model=None, train_dataloader=None, eval_dataloader=None, **kwargs):
        print("GRADIENT EVAL")
        model.to(args.device)
        dataloader = train_dataloader if self.dataset == "train" else eval_dataloader
        all_grads = []
        for batch in tqdm(dataloader):
            batch_grads = compute_sample_grads(batch, model, args.device)
            for i, n in zip(batch["id"], batch_grads):
                N = tuple(map(lambda x: float(x), n))
                elem = (int(i), *N)
                all_grads.append(elem)

        with open(f"{args.output_dir}/grand_{state.epoch}.tsv", "w") as fp:
            fp.write("Id\tScore\n")
            fp.write("\n".join("\t".join(map(str, g)) for g in all_grads))


def compute_el2n(inputs, model, device, reduce_method: str):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        output = model(**batch_input)

    p_intent = nn.functional.softmax(output.intent_logits, dim=1)
    y_intent = nn.functional.one_hot(batch_input["intent_label"], num_classes=model.num_intent)
    err_intent = p_intent - y_intent
    intent_scores = torch.norm(err_intent, dim=1)

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

    return total_scores.detach().cpu().numpy()


class EL2NCallback(TrainerCallback):
    def __init__(self, dataset: str = "train", reduce_method: str = "sum", **kwargs):
        super().__init__(**kwargs)
        assert reduce_method in ["sum", "mean", "norm"], "'reduce_method' must 'sum', 'mean' or 'norm'."
        self.reduce_method = reduce_method
        self.dataset = dataset

    def on_train_end(self, args, state, control, model=None, train_dataloader=None, eval_dataloader=None, **kwargs):
        print("EL2N EVAL")
        model.to(args.device)
        dataloader = train_dataloader if self.dataset == "train" else eval_dataloader
        all_grads = []
        for batch in tqdm(dataloader):
            batch_grads = compute_el2n(batch, model, args.device, self.reduce_method)
            for i, n in zip(batch["id"], batch_grads):
                elem = (int(i), n)
                all_grads.append(elem)

        with open(f"{args.output_dir}/el2n_{state.epoch}.tsv", "w") as fp:
            fp.write("Id\tScore\n") # Intent_Norm\tSlot_Norm\t
            fp.write("\n".join("\t".join(map(str, g)) for g in all_grads))


def compute_persample_loss(inputs, model, device):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        output = model(**batch_input)

    intent_logits = output.intent_logits
    intent_loss_fct = nn.CrossEntropyLoss(reduction="none")
    intent_label = nn.functional.one_hot(batch_input["intent_label"], num_classes=model.num_intent)
    intent_loss = intent_loss_fct(intent_logits.float(), intent_label.float())

    slot_logits = output.slot_logits
    slot_label = batch_input["slot_label"]
    slot_loss_fct = nn.CrossEntropyLoss(reduction="none")
    slot_masks = torch.ne(slot_label, MASK_VALUE).int() * batch_input["attention_mask"]
    slot_label[(1 - slot_masks).bool()] = 0
    slot_label = nn.functional.one_hot(slot_label, num_classes=model.num_slots)
    slot_loss = slot_loss_fct(torch.transpose(slot_logits, 1, 2).float(), torch.transpose(slot_label, 1, 2).float())

    persample_loss = model.alpha * intent_loss + (1 - model.alpha) * torch.sum(slot_masks * slot_loss, dim=1)

    return persample_loss


class PerSampleLossCallback(TrainerCallback):
    def __init__(self, dataset: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_train_end(self, args, state, control, model=None, train_dataloader=None, eval_dataloader=None, **kwargs):
        print("PER-SAMPLE LOSS EVAL")
        model.to(args.device)
        dataloader = train_dataloader if self.dataset == "train" else eval_dataloader
        all_grads = []
        for batch in tqdm(dataloader):
            batch_grads = compute_persample_loss(batch, model, args.device)
            for i, n in zip(batch["id"], batch_grads):
                elem = (int(i), float(n))
                all_grads.append(elem)

        with open(f"{args.output_dir}/loss_{state.epoch}.tsv", "w") as fp:
            fp.write("Id\tScore\n") # Intent_Norm\tSlot_Norm\t
            fp.write("\n".join("\t".join(map(str, g)) for g in all_grads))


class LossCallback(TrainerCallback):
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
    def __init__(self):
        self.start_time = 0
        self.total_time = 0
    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.start_time
        with open(f"{args.output_dir}/time.txt", "a") as fp:
            fp.write("%f\t%f\n" % (state.global_state, step_time))
    def on_train_end(self, args, state, control, **kwargs):
        with open(f"{args.output_dir}/time.txt", "r") as fp:
            times = [tuple(f.strip().split("\t")) for f in fp.readlines()]
        self.total_time = sum([float(t) for _, t in times])
