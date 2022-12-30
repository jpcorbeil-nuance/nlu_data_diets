from dataclasses import dataclass
from typing import *

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.modeling_outputs import BaseModelOutput


MASK_VALUE = -100


class DataCollatorForNLU(DataCollatorForTokenClassification):
    def __init__(self, intent_label: str = "intent_label", slot_label: str = "slot_label", **kwargs):
        super().__init__(**kwargs)
        self.intent_label = intent_label
        self.slot_label = slot_label

    def torch_call(self, features):
        labels = [feature[self.slot_label] for feature in features] if self.slot_label in features[0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[self.slot_label] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[self.slot_label] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


@dataclass
class NLUOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    intent_loss: Optional[torch.FloatTensor] = None
    slot_loss: Optional[torch.FloatTensor] = None
    intent_logits: torch.FloatTensor = None
    slot_logits: torch.FloatTensor = None


class RobertaIntentClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_intent)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaNLUConfig(RobertaConfig):
    def __init__(self,
        num_intent: int = None,
        num_slots: int = None,
        id2intent: dict = None,
        id2slots: dict = None,
        intent_slot_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_intent = num_intent
        self.num_slots = num_slots
        self.id2intent = id2intent
        self.id2slots = id2slots
        self.intent_slot_ratio = intent_slot_ratio


class RobertaForNLU(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_intent = config.num_intent
        self.num_slots = config.num_slots
        self.alpha = config.intent_slot_ratio if config.intent_slot_ratio is not None else 0.5
        self.n_reinit_last_layer = config.n_reinit_last_layer
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if self.n_reinit_last_layer > 0:
            self.reinit_layers()

        self.intent_classifier = RobertaIntentClassificationHead(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slots)

        # Initialize weights and apply final processing
        self.post_init()

    def reinit_layers(self):
        self.roberta.encoder.layer = self.roberta.encoder.layer[:-self.n_reinit_last_layer]
        new_layers = nn.ModuleList([RobertaLayer(self.config) for _ in range(self.n_reinit_last_layer)])
        self.roberta.encoder.layer.extend(new_layers)

    def forward(
        self,
        id,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        intent_label: Optional[torch.LongTensor] = None,
        slot_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], NLUOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        intent_logits = self.intent_classifier(sequence_output)

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = 0
        intent_loss = None
        slot_loss = None
        if intent_label is not None:
            intent_loss_fct = CrossEntropyLoss(reduction="sum")
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent), intent_label.view(-1))
            loss += self.alpha * intent_loss
        if slot_label is not None:
            slot_loss_fct = CrossEntropyLoss(reduction='sum')
            slot_masks = torch.ne(slot_label, MASK_VALUE).int()
            if attention_mask is not None:
                slot_masks *= attention_mask
            active = slot_masks.view(-1) == 1
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slots)[active], slot_label.view(-1)[active])
            loss += (1 - self.alpha) * slot_loss

        if not return_dict:
            output = (intent_logits, slot_logits,) + outputs[2:]
            return ((loss, intent_loss, slot_loss,) + output) if slot_loss is not None else output

        return NLUOutput(
            loss=loss,
            intent_loss=intent_loss,
            slot_loss=slot_loss,
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


def nlu_eval_step(model, batch_input, fullseq_only: bool = False):
    with torch.no_grad():
        outputs = model(**batch_input)

    intent_pred = torch.argmax(outputs.intent_logits, dim=1)
    intent_match = (batch_input["intent_label"] == intent_pred)

    slot_pred = torch.argmax(outputs.slot_logits, dim=2)
    slot_label = batch_input["slot_label"]
    slot_masks = torch.ne(slot_label, MASK_VALUE).int() * batch_input["attention_mask"]

    if not fullseq_only:
        active = slot_masks.view(-1) == 1
        slot_label_active = slot_label.reshape((-1,))[active].detach().cpu().tolist()
        slot_pred_active = slot_pred.reshape((-1,))[active].detach().cpu().tolist()

    slot_match = (slot_label == slot_pred).int()
    slot_all = torch.all(slot_masks*slot_match == slot_masks, dim=1)
    fullseq_match = torch.logical_and(intent_match, slot_all).detach().cpu().tolist()

    if fullseq_only:
        return fullseq_match

    intent_match = intent_match.detach().cpu().tolist()

    return intent_match, fullseq_match, (slot_label_active, slot_pred_active)


def nlu_evaluate(model, testset, device):
    model.to(device)

    intent_tp_tn = 0
    fullseq_tp_tn = 0
    total = 0
    slot_y = []
    slot_preds = []
    with torch.no_grad():
        for inputs in tqdm(testset, desc="Final Test Eval"):
            batch_input = {k: inputs[k].to(device) for k in inputs}
            outputs = model(**batch_input)

            intent_match, fullseq_match, (slot_label, slot_pred) = nlu_eval_step(model, batch_input)

            intent_tp_tn += torch.sum(intent_match).detach().cpu().numpy()

            slot_y.extend(slot_label)
            slot_preds.extend(slot_pred)

            fullseq_tp_tn += torch.sum(fullseq_match).detach().cpu().numpy()

            total += len(intent_pred)

    slot_f1 = f1_score(slot_y, slot_preds, average="micro")
    output = {"intent_acc": intent_tp_tn/total, "slot_f1": slot_f1, "fullseq_acc": fullseq_tp_tn/total}
    output = {k: float(v) for k, v in output.items()}
    return output
