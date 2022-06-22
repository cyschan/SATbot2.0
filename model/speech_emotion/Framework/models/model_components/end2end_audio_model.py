import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.utils.model_utils import same_padding_1d


class End2EndAudioModel(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(End2EndAudioModel, self).__init__()

        # 64 output channels, kernel size of 8
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, 9, padding=same_padding_1d(in_channels, 9, 1))
        self.pool1 = nn.MaxPool1d(10, 10)

        # 128 output channels, kernel size of 6
        self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 2, 7, padding=same_padding_1d(out_channels // 4, 7, 1))
        self.pool2 = nn.MaxPool1d(10, 10)

        # 256 output channels, kernel size of 6
        self.conv3 = nn.Conv1d(out_channels // 2, out_channels, 7, padding=same_padding_1d(out_channels // 2, 7, 1))
        self.pool3 = nn.MaxPool1d(10, 10)

    def forward(self, input):
        input = input.transpose(1, 2)
        out = F.relu(self.conv1(input))
        out = self.pool1(out)

        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = out.transpose(1, 2)
        return out
