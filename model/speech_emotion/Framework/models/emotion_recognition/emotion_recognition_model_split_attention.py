import torch
import torch.nn as nn
import torch.nn.functional as F

from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.models.model_components.lstm_cells import ConversationLSTMStack
from model.speech_emotion.Framework.models.model_components.bert_nlp_model import BERTModel
from model.speech_emotion.Framework.models.model_components.end2end_audio_model import End2EndAudioModel
from model.speech_emotion.Framework.models.model_components.end2end_bilstm_model import End2EndBiLSTMModel


class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, number_of_emotions, num_layers_bilstm, cell_state_clipping=100,
                 bias=True, freeze_bert=True, predict_emotion=True, predict_details=True, normalisation_tuple=None):
        super(EmotionRecognitionModel, self).__init__()

        """                                       CONFIGURE MODEL PARAMETERS                                         """
        self.hidden_size = hidden_size
        self.normalisation_tuple = normalisation_tuple
        self.enable_cuda = True
        self.device = get_gpu_device()

        """                                    CONFIGURE NN MODULE PARAMETERS                                        """
        """                   A LOT OF THE DECLARATIONS HERE ARE REDUNDANT BUT ADDED FOR CLARITY                     """
        # Should the model predict emotion/details/both
        self.predict_emotion = predict_emotion
        self.predict_details = predict_details

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
        self.emotion_attention_layer = nn.Linear(hidden_size, 1)
        self.detail_attention_layer = nn.Linear(hidden_size, 1)

        # This is an LSTM used on the hidden state of emotion prediction, the idea is that it will keep memory of how
        # the speaker's emotions have changed over time and this will likely affect future predictions.
        # The input to this LSTM is the output of self.emotion_layer before the final emotion prediction.
        self.emotion_memory = ConversationLSTMStack(hidden_size, hidden_size, num_layers=1)

        # Final output heads of the model
        if predict_emotion:
            self.emotion_layer = nn.Linear(hidden_size, hidden_size)
            self.emotion_head = nn.Linear(hidden_size, number_of_emotions)

        # Arousal, valence, and dominance have 3 possible values for negative, neutral, and positive
        if predict_details:
            self.arousal_layer = nn.Linear(hidden_size, hidden_size)
            self.arousal_head = nn.Linear(hidden_size, 3)
            self.valence_layer = nn.Linear(hidden_size, hidden_size)
            self.valence_head = nn.Linear(hidden_size, 3)
            self.dominance_layer = nn.Linear(hidden_size, hidden_size)
            self.dominance_head = nn.Linear(hidden_size, 3)

    def forward(self, audio, token_ids, attention_mask, participants, hx=None):
        audio = self.normalise_audio(audio)

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

        emotion_attention = self.emotion_attention_layer(joined_tensor)
        emotion_attention = torch.softmax(emotion_attention, 1)

        emotion_out = torch.mul(joined_tensor, emotion_attention)
        emotion_out = torch.sum(emotion_out, dim=1)

        detail_attention = self.detail_attention_layer(joined_tensor)
        detail_attention = torch.softmax(detail_attention, 1)

        detail_out = torch.mul(joined_tensor, detail_attention)
        detail_out = torch.sum(detail_out, dim=1)

        # Store only the outputs that are required based on the predict_emotion and predict_details parameters
        return_vals = []

        if self.predict_emotion:
            # Assuming entire batch is the same conversation i.e. batch size = 1
            # so make shape (1, batch size, hidden size)
            batch = emotion_out.size(0)
            hidden = emotion_out.size(1)
            emotion = emotion_out.view(1, batch, hidden)
            emotion, hx = self.emotion_memory(emotion, participants, hx)
            emotion = emotion.view(batch, hidden)
            emotion = self.emotion_head(emotion)
            return_vals.append(emotion)

        if self.predict_details:
            arousal = self.arousal_layer(detail_out)
            arousal = self.arousal_head(arousal)
            valence = self.arousal_layer(detail_out)
            valence = self.valence_head(valence)
            dominance = self.arousal_layer(detail_out)
            dominance = self.dominance_head(dominance)
            return_vals.append(arousal)
            return_vals.append(valence)
            return_vals.append(dominance)

        # Return tuple of predictions, the attention layer
        attention = {'audio bilstm attention': bilstm_attn.detach(), 'audio nlp attention': emotion_attention.detach(),
                     'detail nlp attention': detail_attention.detach()}
        out = {'emotion out': emotion_out, 'detail out': detail_out}
        return tuple(return_vals), attention, out, hx

    def normalise_audio(self, audio):
        with torch.no_grad():
            audio_mean, audio_std = self.normalisation_tuple
            audio = (audio - audio_mean) / audio_std

        return audio