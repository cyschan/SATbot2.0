from model.speech_emotion.Framework.models.emotion_recognition.emotion_recognition_model import EmotionRecognitionModel
from model.speech_emotion.Framework.utils.model_utils import load_model
from transformers import BertTokenizer
import torch

emotion_parameters = {
    'input_size': 1, 'hidden_size': 1024,
    'number_of_emotions': len(['ang', 'hap', 'sad', 'sur', 'neu', 'fru', 'exc', 'fea']), 'num_layers_bilstm': 2,
    'cell_state_clipping': None, 'bias': True, 'freeze_bert': False, 'predict_emotion': True, 'predict_details': True,
    'normalisation_tuple': (-8.541886477750582e-06, 0.05738852859500303)
}

emotion_predictions = {'ang': 0, 'hap': 1, 'sad': 2, 'sur': 3, 'neu': 4, 'fru':5 , 'exc': 6, 'fea': 7}


def get_best_model(model_file):
    model = EmotionRecognitionModel(**emotion_parameters)
    load_model(model, model_file)
    return model, emotion_predictions

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-large-uncased')
