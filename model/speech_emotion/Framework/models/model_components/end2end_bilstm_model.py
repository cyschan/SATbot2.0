import torch
import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.models.model_components.lstm_cells import LSTMStack


class End2EndBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, cell_state_clipping=100, bias=True):
        super(End2EndBiLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.enable_cuda = True
        self.device = get_gpu_device()

        # LSTM Stack
        if cell_state_clipping == None:
            # print('USING PYTORCH STACK')
            self.lstm_stack = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first=False, dropout=0, bidirectional=True).to(self.device)
            self.use_pytorch = True
        else:
            self.lstm_stack = LSTMStack(input_size, hidden_size, num_layers, cell_state_clipping, bias, bidirectional=True).to(self.device)
            self.use_pytorch = False

        # Attention will be used on the sequences provided in the audio
        self.attention_layer = nn.Linear(hidden_size, 1)

        # Output
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, states_0=None):
        # Outputs shape: (sequence length, batch size, 2 * hidden size)
        # Pytorch LSTMs expect input to be (sequence, batch size, input size)
        if self.use_pytorch:
            input = input.transpose(0, 1)

        outputs, states = self.lstm_stack(input, states_0)

        # Split outputs into each direction
        shaped_outputs = outputs.view(outputs.size(0), outputs.size(1), 2, self.hidden_size)
        fwd_outputs = shaped_outputs[:, :, 0, :].squeeze(dim=2)
        bwd_outputs = shaped_outputs[:, :, 1, :].squeeze(dim=2)

        # Size will now be (sequence length, batch size, hidden size)
        outputs = fwd_outputs + bwd_outputs
        outputs = outputs.to(self.device)

        # Transpose so that batch dimension is first
        outputs = outputs.transpose(0, 1)

        # attention shape: (batch size, sequence length, 1)
        attention = self.attention_layer(outputs)
        attention = torch.softmax(attention, 1)

        out = torch.mul(outputs, attention)
        out = torch.sum(out, dim=1)

        # Shape of out should now be (batch size, hidden size) so same output layer can be used
        mean_prediction = self.output_layer(out)

        # Return mean_prediction and the attention output so that it can be visualised at a later date
        return mean_prediction, attention
