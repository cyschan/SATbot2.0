import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.utils.model_utils import same_padding_1d


class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_state_clipping=100, bias=True, bidirectional=False):
        super(LSTMStack, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.enable_cuda = True
        self.device = get_gpu_device()

        self.cells = nn.ModuleList()
        if bidirectional:
            self.reverse_cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(CustomLSTMCell(cell_input_size, hidden_size, cell_state_clipping, bias))

    def forward(self, input, states_0=None):
        # Input should be of shape (batch, sequences, input_size)
        # Initialise hidden states
        if states_0 is None:
            states_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
            states_0 = (states_0, states_0)

        initial_hidden_states, initial_cell_states = states_0
        states = []
        for layer in range(self.num_layers):
            states.append((initial_hidden_states[layer, :, :], initial_cell_states[layer, :, :]))

        # Outputs of LSTM stack should be of shape (sequence length, batch size, hidden size)
        output_size = self.hidden_size
        if self.bidirectional:
            output_size = output_size * 2
        outputs = torch.empty(input.size(1), input.size(0), output_size).to(self.device)

        if self.bidirectional:
            states_rev = []
            for layer in range(self.num_layers):
                states_rev.append((initial_hidden_states[layer, :, :], initial_cell_states[layer, :, :]))

        for sequence in range(input.size(1)):
            for layer in range(self.num_layers):
                # Input to LSTM is either the original input, or the previous layer's hidden state
                cell_input = input[:, sequence, :] if layer == 0 else states[layer - 1][0]
                hidden, cell = self.cells[layer](cell_input, states[layer])
                states[layer] = (hidden, cell)

                if self.bidirectional:
                    rev_cell_input = input[:, -sequence - 1, :] if layer == 0 else states_rev[layer - 1][0]
                    hidden_rev, cell_rev = self.cells[layer](rev_cell_input, states_rev[layer])
                    states_rev[layer] = (hidden_rev, cell_rev)

            # Outputs are the final layer's hidden state
            outputs[sequence, :, :self.hidden_size] = hidden
            if self.bidirectional:
                outputs[-sequence - 1, :, self.hidden_size:] = hidden_rev

        # Returns the final hidden state in each layer in list as output along with the last hidden/cell states for each
        # layer
        # Stack the outputs into one tensor
        if self.bidirectional:
            stacked_hidden_states = torch.zeros((2 * self.num_layers, input.size(0), self.hidden_size)).to(self.device)
            stacked_cell_states = torch.zeros((2 * self.num_layers, input.size(0), self.hidden_size)).to(self.device)
        else:
            stacked_hidden_states = torch.zeros((self.num_layers, input.size(0), self.hidden_size)).to(self.device)
            stacked_cell_states = torch.zeros((self.num_layers, input.size(0), self.hidden_size)).to(self.device)

        for i in range(len(states)):
            stacked_hidden_states[i, :, :] = states[i][0]
            stacked_cell_states[i, :, :] = states[i][1]
            if self.bidirectional:
                stacked_hidden_states[i + len(states), :, :] = states_rev[i][0]
                stacked_cell_states[i + len(states), :, :] = states_rev[i][1]

        return outputs, (stacked_hidden_states, stacked_cell_states)


class ConversationLSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_state_clipping=100, bias=True, bidirectional=False):
        super(ConversationLSTMStack, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.enable_cuda = True
        self.device = get_gpu_device()

        self.cells = nn.ModuleList()
        if bidirectional:
            self.reverse_cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(CustomLSTMCell(cell_input_size, hidden_size, cell_state_clipping, bias))

    def forward(self, input, participants, states_0=None):
        # Input should be of shape (1, batch, input_size)
        # Participants should be list of speakers of length same as batch size
        # Initialise hidden states
        if states_0 is None:
            states_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
            states_0 = (None, (states_0, states_0))

        zero_tensor_state = Variable(torch.zeros(input.size(0), self.hidden_size))

        last_speaker, (initial_hidden_states, initial_cell_states) = states_0
        states = []
        for layer in range(self.num_layers):
            # We don't save hidden states so use 0 state
            states.append((zero_tensor_state.detach().clone().to(self.device), initial_cell_states[layer, :, :]))

        # Outputs of LSTM stack should be of shape (sequence length, batch size, hidden size)
        output_size = self.hidden_size

        if self.bidirectional:
            output_size = output_size * 2

        outputs = torch.empty(input.size(1), input.size(0), output_size).to(self.device)

        if self.bidirectional:
            raise ValueError('Not implemented correctly for conversational use')
            # states_rev = []
            # for layer in range(self.num_layers):
            #     states_rev.append((initial_hidden_states[layer, :, :], initial_cell_states[layer, :, :]))

        for sequence in range(input.size(1)):
            for layer in range(self.num_layers):
                # Input to LSTM is either the original input, or the previous layer's hidden state
                cell_input = input[:, sequence, :] if layer == 0 else states[layer - 1][0]

                # If speaker has changed then reset the states
                # print('Last sequence had speaker', last_speaker)
                speaker = participants[sequence]
                # print('Current sequence has speaker', speaker)
                if speaker != last_speaker:
                    states[layer] = (zero_tensor_state.detach().clone().to(self.device),
                                     zero_tensor_state.detach().clone().to(self.device))
                    # print('Different speaker, erasing states to', states[layer])
                # else:
                #     print('Same speaker, keeping states as', states[layer])
                last_speaker = speaker

                # Since we are wanting to model long term memory of the speaker we forget the hidden state/short term
                input_state = (zero_tensor_state.detach().clone().to(self.device), states[layer][1])
                # print('inputting state without hidden state', input_state)

                hidden, cell = self.cells[layer](cell_input, input_state)

                states[layer] = (hidden, cell)
                # print('Saving states', states[layer])

                if self.bidirectional:
                    raise ValueError('Not implemented correctly for conversational use')
                    # rev_cell_input = input[:, -sequence - 1, :] if layer == 0 else states_rev[layer - 1][0]
                    # hidden_rev, cell_rev = self.cells[layer](rev_cell_input, states_rev[layer])
                    # hidden_rev = None
                    # states_rev[layer] = (hidden_rev, cell_rev)

            # Outputs are the final layer's hidden state
            outputs[sequence, :, :self.hidden_size] = hidden
            # if self.bidirectional:
            #     outputs[-sequence - 1, :, self.hidden_size:] = hidden_rev

        # Returns the final hidden state in each layer in list as output along with the last hidden/cell states for each
        # layer
        # Stack the outputs into one tensor
        # if self.bidirectional:
        #     stacked_hidden_states = torch.zeros((2 * self.num_layers, input.size(0), self.hidden_size)).to(self.device)
        #     stacked_cell_states = torch.zeros((2 * self.num_layers, input.size(0), self.hidden_size)).to(self.device)
        # else:
        stacked_hidden_states = torch.zeros((self.num_layers, input.size(0), self.hidden_size)).to(self.device)
        stacked_cell_states = torch.zeros((self.num_layers, input.size(0), self.hidden_size)).to(self.device)

        for i in range(len(states)):
            stacked_hidden_states[i, :, :] = states[i][0]
            stacked_cell_states[i, :, :] = states[i][1]
            # if self.bidirectional:
            #     stacked_hidden_states[i + len(states), :, :] = states_rev[i][0]
            #     stacked_cell_states[i + len(states), :, :] = states_rev[i][1]

        return outputs, (last_speaker, (stacked_hidden_states, stacked_cell_states))


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_state_clipping, bias):
        super(CustomLSTMCell, self).__init__()
        # cell_constructor could be changed by an argument to allow for different cells e.g. a custom implementation
        # of peephole cells in pytorch could be used
        cell_constructor = nn.LSTMCell
        self.cell = cell_constructor(input_size, hidden_size, bias)
        self.cell_state_clipping = cell_state_clipping

    def forward(self, input, h0):
        # h0 is a tuple of the initial (hidden state, cell state)
        h1, c1 = self.cell(input, h0)
        if self.cell_state_clipping:
            torch.clamp(c1, min=-self.cell_state_clipping, max=self.cell_state_clipping)
        return h1, c1