import torch
import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.global_config import get_gpu_device


class WeightedHead(nn.Module):
    def __init__(self, input_size):
        super(WeightedHead, self).__init__()

        """                                       CONFIGURE MODEL PARAMETERS                                         """
        self.enable_cuda = True
        self.device = get_gpu_device()

        """                                    CONFIGURE NN MODULE PARAMETERS                                        """
        """                   A LOT OF THE DECLARATIONS HERE ARE REDUNDANT BUT ADDED FOR CLARITY                     """
        # Add a couple of linear layers to allow each head to adequately learn prediction specific features
        self.linear_1 = nn.Linear(input_size, 3 * input_size // 4)
        self.linear_2 = nn.Linear(3 * input_size // 4, input_size // 2)
        self.utterance_predictions = nn.Linear(input_size // 2, 1)
        self.utterance_weights = nn.Linear(input_size // 2, 1)

    def forward(self, hidden_state):
        head_state = F.relu(self.linear_1(hidden_state))
        head_state = F.relu(self.linear_2(head_state))

        utterance_level_predictions = self.utterance_predictions(head_state)
        utterance_level_weights = self.utterance_weights(head_state)

        utterance_level_weights = torch.softmax(utterance_level_weights, dim=0)

        prediction = torch.mul(utterance_level_weights, utterance_level_predictions)
        prediction = torch.sum(prediction, dim=0)
        return prediction, utterance_level_predictions, utterance_level_weights