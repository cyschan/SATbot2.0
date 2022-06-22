import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERTModel(nn.Module):
    def __init__(self, freeze_bert=True, bert_model=BertModel, pre_trained_bert='bert-large-uncased'):
        super(BERTModel, self).__init__()
        self.bert_model = bert_model.from_pretrained(pre_trained_bert)
        if freeze_bert:
            for p in self.bert_model.parameters():
                p.requires_grad = False

    def get_output_size(self):
        return self.bert_model.config.hidden_size

    def forward(self, audio, token_ids, attention_mask, output_cls=True):
        bert_out = self.bert_model(token_ids, attention_mask=attention_mask)
        if type(bert_out) is tuple:
            bert_out = bert_out[0]

        if output_cls:
            return bert_out[:, 0]

        return bert_out