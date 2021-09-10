from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from torch import nn
from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

class RobertaSpanPredictor(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(RobertaSpanPredictor, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, 3)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids)
        outputs = outputs[0]
        outputs = outputs.reshape(-1, outputs.size()[-1])

        out = self.act(self.linear1(self.dropout(outputs)))
        logits = self.linear2(out).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        return logits