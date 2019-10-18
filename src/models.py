import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForBinarySequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForBinarySequenceClassification, self).__init__(config)
        self.num_labels = self.config.num_labels
        if self.num_labels != 1:
            raise Exception("Binary classification requires one label to apply BCE.")

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, self.config.num_labels)

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
        logits = self.output_layer(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
