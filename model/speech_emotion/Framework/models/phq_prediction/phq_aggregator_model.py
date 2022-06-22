import torch
import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.models.phq_prediction.weighted_head import WeightedHead


class AggregatorModel(nn.Module):
    def __init__(self, phq_hidden_size, phq_shrink_hidden_size, emotion_logit_length=None):
        super(AggregatorModel, self).__init__()

        """                                       CONFIGURE MODEL PARAMETERS                                         """
        self.phq_hidden_size = phq_hidden_size
        self.phq_shrink_hidden_size = phq_shrink_hidden_size
        if emotion_logit_length is None:
            self.joined_tensor_size = phq_shrink_hidden_size
        else:
            self.joined_tensor_size = phq_shrink_hidden_size + emotion_logit_length

        self.enable_cuda = True
        self.device = get_gpu_device()

        """                                       CONFIGURE NN MODULES                                               """
        self.shrink_hidden_states = nn.Linear(self.phq_hidden_size, self.phq_shrink_hidden_size)

        self.linear_layer_1 = nn.Linear(self.joined_tensor_size, self.joined_tensor_size)
        self.linear_layer_2 = nn.Linear(self.joined_tensor_size, self.joined_tensor_size)

        # Heads should be phq and ptsd prediction unless changed for fine tuning
        phq_head = WeightedHead(self.joined_tensor_size)
        ptsd_head = WeightedHead(self.joined_tensor_size)
        self.heads = nn.ModuleList([phq_head, ptsd_head])

    def setup_fine_tune_for_gad_7(self):
        # First add the GAD 7 head
        self.heads.append(WeightedHead(self.joined_tensor_size))

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze GAD 7 head
        for param in self.heads[-1].parameters():
            param.requires_grad = True

        # Unfreeze linear layer 2 as well
        # self.linear_layer_2.weight.requires_grad = True
        # self.linear_layer_2.bias.requires_grad = True

    def forward(self, hidden_states, emotion_logits=None):
        hidden_states = self.shrink_hidden_states(hidden_states)

        if emotion_logits is not None:
            hidden_states = torch.cat((hidden_states, emotion_logits), 1)

        # relu since predictions are 0 - X of some upper bound
        hidden_states = F.relu(self.linear_layer_1(hidden_states))
        hidden_states = F.relu(self.linear_layer_2(hidden_states))

        predictions = []
        utterance_predictions = []
        utterance_weights = []

        for i in range(len(self.heads)):
            pred, utt_pred, utt_weight = self.heads[i](hidden_states)
            predictions.append(pred)
            utterance_predictions.append(utt_pred)
            utterance_weights.append(utt_weight)

        return predictions, utterance_predictions, utterance_weights
