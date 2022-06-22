import numpy as np
import scipy.signal as sc
import wavio
import os
import os.path
import librosa
import speech_recognition as sr
from model.speech_emotion.Framework.emotion_recognition_tool.emotion_recognition_tool import EmotionRecognition
from model.speech_emotion.Framework.emotion_recognition_tool.best_model_config import get_tokenizer
from model.speech_emotion.Framework.models.emotion_recognition.emotion_recognition_model import EmotionRecognitionModel
from model.speech_emotion.Framework.utils.model_utils import load_model
from model.speech_emotion.Framework.global_config import set_cpu_only
import torch

class SpeechEmotionAnalyser:
    def __init__(self):
        set_cpu_only()
        #keep the order of emotions, but you can rename them to any other form of the word e.g. happiness, do not remove any
        self.emotions_to_predict = ['Happy', 'Sad', 'Angry', 'Surprised', 'Disgust', 'Fear', 'other']
        self.model = self.get_model(self.emotions_to_predict, model_path='/home/ccys/SATbot2.0/model/speech_emotion/multilabel7.pth', model_type='multilabel')
        #thresholds set during validation
        self.thresholds = [0.4, 0.25, 0.2, 0.1, 0.25, 0.1, 0.1]

    def get_emotion(self, recording, text):  
        model, audio, token_ids, attention_mask = self.load_data(model=self.model, audiofile=recording, text=text, model_type='multilabel')
        if audio is None:
            return 'Noemo'
        else:
            prediction = self.predict(model, audio, token_ids, attention_mask)
            pred_emotion = self.process_pred(prediction, self.emotions_to_predict, thresholds=self.thresholds, model_type='multilabel')
            return pred_emotion # return string
    """
    Helper functions to run per utterance emotion recognition.
    @author Lucia Simkanin
    """

    """
    If multilabel creates and loades the model, if mutliclass creates an object which holds the model and additional 
    variables/functions as created by James Tavernor.
    @param emotions emotions to be predicted 
    @param model_path path to the model
    @param model_type multiclass or multilabel classification
    @return model returns the loaded model/object
    """
    def get_model(self, emotions, model_path='/home/ccys/SATbot2.0/model/speech_emotion/Framework/emotion_recognition_tool/best_model_cpu.pth', model_type='multiclass'):
        if model_type == 'multiclass':
            model = EmotionRecognition(model_path, emotions=emotions)
        if model_type == 'multilabel':
            model = EmotionRecognitionModel(input_size=1, hidden_size=1024, number_of_emotions=len(emotions),
                                            num_layers_bilstm=2, cell_state_clipping=None, bias=True, freeze_bert=False,
                                            predict_emotion=True, predict_details=False, normalisation_tuple=(-0.0009942063985169049, 0.08078398772794018))

        #load_model(model, model_path)
        return model

    """
    Takes care of pre-processing the audio and text data, including transcription, tokenization and encoding, as well as
    loading the data into tensors.
    @param model the model received from get_model function
    @param audiofile a dat file that stores the audio data in the form of a binary array 
    @param model_type multiclass or multilabel classification
    @return model
    @return audio, token_ids, attention_mask input to the emotion recognition model
    """
    def load_data(self, model, audiofile, text = None, model_type='multiclass'):
        #load the data to np array and resample it to 48000 (same rate as training audio)
        if text == None:
            audio_data = np.fromfile(audiofile, dtype=np.float32)
            number_of_samples = round(len(audio_data) * float(48000) / 44100)
            audio_data = sc.resample(audio_data, number_of_samples)
            #transform the array to a wav file/raw input
            filename = 'test'
            index = str(0)
            audiofile = filename + index + '.wav'
            wavio.write(audiofile, audio_data, 48000, sampwidth=2)
            #speech recognition to get the text
            r = sr.Recognizer()
            with sr.AudioFile(audiofile) as source:
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data)
                except sr.UnknownValueError:
                    text = 'no_text_detected'
            #if no text is detected return None as audio etc.
            if text == 'no_text_detected':
                os.remove(audiofile)
                return model, None, None, None
            #text preprocessing and audio/text loading
        if model_type == 'multiclass':
            audio = model.load_audio(audiofile).unsqueeze(dim=0)
            tokens = model.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.tensor(model.tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(dim=0)
        if model_type == 'multilabel':
            tokenizer = get_tokenizer()
            audio = librosa.core.load(audiofile, sr=48000, mono=True)
            audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
            audio = torch.from_numpy(audio).float().view(-1, model.input_size).unsqueeze(dim=0)
            tokens = tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(dim=0)
        #os.remove(audiofile)
        attention_mask = torch.tensor([1] * len(tokens)).unsqueeze(dim=0)
        return model, audio, token_ids, attention_mask


    """
    Processes the predicted emotions, to only output one emotion in the form of a string.
    @prediction predicted emotions, of different form based on multiclass/multilabel model
    @emotions emotions to predict in the correct order
    @thresholds for detecting the emotion for multilabel model, set during validation 
    @model_type multilabel or multiclass classification
    @return pred_emotion the most dominant emotion as a string
    """
    def process_pred(self, prediction, emotions, thresholds=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], model_type='multiclass'):
        if model_type == 'multiclass':
            emotion, _, _, _, _ = prediction
            _, pred_emotion, _, _ = emotion
            pred_emotion = pred_emotion[0]
        if model_type == 'multilabel':
            #find out which emotion has the largest difference between the value and its threshold
            pred_emotion = prediction - thresholds
            #if multiple emotions detected with the same difference, they will be selected based on the order of emotions
            index = np.argmax(pred_emotion)
            #if the largest difference is negative, it means the emotion nor any other were detected
            if prediction[index] - thresholds[index] < 0:
                pred_emotion = 'Noemo'
            else:
                pred_emotion = emotions[index]
        return pred_emotion

    """
    Used to acquire the predicted emotions from the multilabel model.
    @param model
    @param audio
    @param toke_ids
    @param attention_mask
    @return pred_emotions a numpy array consisting of the 0-1 values for the predicted emotions
    """
    def predict(self, model, audio, token_ids, attention_mask):
        model.eval()
        with torch.no_grad():
            predictions, attention, hidden_out, hx = model(audio, token_ids, attention_mask)
        pred_emotions = predictions[0]
        pred_emotions = torch.sigmoid(pred_emotions).numpy()[0]
        return pred_emotions