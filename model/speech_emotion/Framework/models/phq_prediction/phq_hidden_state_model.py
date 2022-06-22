import torch
import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.models.model_components.lstm_cells import ConversationLSTMStack
from model.speech_emotion.Framework.models.model_components.bert_nlp_model import BERTModel
from model.speech_emotion.Framework.models.model_components.end2end_audio_model import End2EndAudioModel
from model.speech_emotion.Framework.models.model_components.end2end_bilstm_model import End2EndBiLSTMModel


class HiddenStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers_bilstm, cell_state_clipping=100, bias=True, freeze_bert=True,
                 predict_phq_8=True, predict_ptsd_severity=True, normalisation_tuple=None):
        super(HiddenStateModel, self).__init__()

        """                                       CONFIGURE MODEL PARAMETERS                                         """
        self.hidden_size = hidden_size
        self.normalisation_tuple = normalisation_tuple
        self.enable_cuda = True
        self.device = get_gpu_device()

        """                                    CONFIGURE NN MODULE PARAMETERS                                        """
        """                   A LOT OF THE DECLARATIONS HERE ARE REDUNDANT BUT ADDED FOR CLARITY                     """
        # Should the model predict emotion/details/both
        self.predict_phq_8 = predict_phq_8
        self.predict_ptsd_severity = predict_ptsd_severity

        # Audio CNN Parameters
        # Define sizes of vectors throughout
        audio_input_size = input_size
        audio_output_size = hidden_size

        # Audio BiLSTM Parameters
        # Output of audio goes into the bilstm so we have already got the input size from above
        # num_layers_bilstm/cell_state_clipping/bias is passed as a parameter to the emotion recognition model
        bilstm_hidden_size = hidden_size
        bilstm_output_size = hidden_size

        """                                       CONFIGURE NN MODULES                                               """
        # Define audio and bilstm models used on the audio
        self.audio_model = End2EndAudioModel(audio_input_size, audio_output_size)
        self.bilstm_model = End2EndBiLSTMModel(audio_output_size, bilstm_hidden_size, bilstm_output_size,
                                               num_layers_bilstm, cell_state_clipping, bias)

        # Define the NLP model used on the text transcript
        self.nlp_model = BERTModel(freeze_bert)
        self.nlp_linear_layer = nn.Linear(self.nlp_model.get_output_size(), hidden_size)

        # Define attention layer used towards end of model to choose between NLP and Audio output vectors
        self.attention_layer = nn.Linear(hidden_size, 1)

        # This is an LSTM used on the hidden state of emotion prediction, the idea is that it will keep memory of how
        # the speaker's emotions have changed over time and this will likely affect future predictions.
        # The input to this LSTM is the output of self.emotion_layer before the final emotion prediction.
        self.speaker_memory = ConversationLSTMStack(hidden_size, hidden_size, num_layers=1)

    def forward(self, audio, token_ids, attention_mask, participants, hx=None):
        audio = self.normalise_audio(audio)
        batch_size = audio.size(0)

        # Audio Model
        # Extract features with CNN and then apply BiLSTM with attention to these features
        audio_out = self.audio_model(audio)
        audio_out, bilstm_attn = self.bilstm_model(audio_out)

        # NLP Model
        # Apply NLP model and then pass through a linear layer to ensure it is same size as the audio_output
        nlp_out = self.nlp_model(audio, token_ids, attention_mask)
        nlp_out = F.relu(self.nlp_linear_layer(nlp_out))

        # Apply Attention on Outputs
        # first join outputs together, they should both be (batch size, hidden size) so easy to combine
        joined_tensor = torch.cat((audio_out.unsqueeze(dim=1), nlp_out.unsqueeze(dim=1)), dim=1)

        attention = self.attention_layer(joined_tensor)
        attention = torch.softmax(attention, 1)

        out = torch.mul(joined_tensor, attention)
        out = torch.sum(out, dim=1)

        hidden = out.size(1)
        speaker_out = out.view(1, batch_size, hidden)
        speaker_out, hx = self.speaker_memory(speaker_out, participants, hx)
        speaker_out = speaker_out.view(batch_size, hidden)

        # Return tuple of predictions, the attention layer
        attention = {'audio bilstm attention': bilstm_attn.detach(), 'audio nlp attention': attention.detach()}
        return speaker_out, attention, out, hx

    def normalise_audio(self, audio):
        with torch.no_grad():
            audio_mean, audio_std = self.normalisation_tuple
            audio = (audio - audio_mean) / audio_std

        return audio