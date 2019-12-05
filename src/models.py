import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class LoadParallellMixin(nn.Module):
    def load_parallel_state_dict(self, to_load, remove_prefix="module."):
        """
        Supports loading parallel and non parallel state dicts into non parallel models.

        We need this due to apex needing to be initialized on the non parallelized model.
        """
        if not any(k.startswith(remove_prefix) for k in self.state_dict().keys()):
            self.load_state_dict(
                {
                    k[len(remove_prefix):]: v for k, v in to_load.items()
                    if k.startswith(remove_prefix) }
            )
        else:
            self.load_state_dict(to_load)


class BertForBinarySequenceClassification(BertPreTrainedModel, LoadParallellMixin):
    def __init__(self, config, positive_class_weight, **kwargs):
        super(BertForBinarySequenceClassification, self).__init__(config, **kwargs)
        self.num_labels = self.config.num_labels
        self.positive_class_weight = positive_class_weight
        if self.num_labels != 1:
            raise Exception("Binary classification requires one label to apply BCE.")

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(weight=self.positive_class_weight)
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForBinaryTokenClassification(BertPreTrainedModel, LoadParallellMixin):
    def __init__(self, config, positive_class_weight, **kwargs):
        super(BertForBinaryTokenClassification, self).__init__(config, **kwargs)
        self.positive_class_weight = positive_class_weight
        self.num_labels = config.num_labels
        if self.num_labels != 1:
            raise Exception("Binary classification requires one label to apply BCE.")

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(weight=self.positive_class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
