import torch
import librosa
import numpy as np

from model.speech_emotion.Framework.emotion_recognition_tool.best_model_config import get_best_model, get_tokenizer
from model.speech_emotion.Framework.global_config import get_gpu_device


class EmotionRecognition:
    def __init__(self, emotion_model_file, emotions=None, min_confidence=None, min_length=None, max_length=None):
        self.full_names = ['angry', 'happy', 'sad', 'surprised', 'neutral', 'frustrated', 'excited', 'fearful']
        self.aro_val_dom_names = ['negative', 'neutral', 'positive']
        self.device = get_gpu_device()

        if emotions is None:
            self.emotions = ['angry', 'happy', 'sad']
        else:
            self.emotions = emotions

        self.emotion_recognition_model, emotion_to_vector_position = get_best_model(emotion_model_file)
        self.emotion_recognition_model.eval()
        for param in self.emotion_recognition_model.parameters():
            param.requires_grad = False
        self.emotion_recognition_model.to(self.device)

        self.tokenizer = get_tokenizer()

        # Emotion memory is important, this should be reset to None if we want to model a new speaker
        self.conversational_memory = None

        # Construct the index tuple for removing unwanted emotion predictions, validation can be done simultaneously
        tuple_values = []

        for emotion in self.emotions:
            if emotion not in self.full_names:
                valid_emotions = self.full_names
                msg = 'Invalid emotion {}, emotions must be a list of emotions from {}.'.format(emotion, valid_emotions)
                raise ValueError(msg)

            tuple_values.append(emotion_to_vector_position[emotion[:3]])

        self.tuple_of_idx = tuple(tuple_values)
        self.input_size = self.emotion_recognition_model.input_size
        self.min_confidence = min_confidence
        self.min_length = min_length
        self.max_length = max_length
        self.utt_count = 1

    def __call__(self, audio_path):
        # Split audio and get transcript before calling predict
        pass

    def load_audio(self, audio_path, start_time=None, end_time=None, mono=True):
        if start_time is None and end_time is None:
            audio = librosa.core.load(audio_path, sr=16000, mono=mono)
        else:
            length = end_time - start_time
            audio = librosa.core.load(audio_path, sr=48000, mono=mono, offset=start_time, duration=length)
            # Log these utterances for the demo
            librosa.output.write_wav('files/utterance_{}.wav'.format(self.utt_count), audio[0], 48000)
            self.utt_count += 1
        librosa.output.write_wav('/home/ccys/SATbot2.0/model/speech_emotion/utterance16000.wav', audio[0], 16000)
        #audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
        #resample = librosa.resample(audio[0], orig_sr=48000, target_sr=16000)
        #librosa.output.write_wav('/home/ccys/SATbot2.0/model/speech_emotion/utterance16000.wav', resample[0], 16000)
        #audio = librosa.resample(audio[0], orig_sr=48000, target_sr=48000).astype(np.float32)
        audio = audio[0].astype(np.float32)
        
        return torch.from_numpy(audio).float().to(self.device).view(-1, self.input_size)

    def check_transcript_valid(self, confidence, length):
        # Each is valid if there is no requirement set, or value is at least the minimum/maximum
        confidence_valid = self.min_confidence is None or self.min_confidence <= confidence
        min_length_valid = self.min_length is None or self.min_length <= length
        max_length_valid = self.max_length is None or self.max_length >= length

        return confidence_valid and min_length_valid and max_length_valid

    def load_transcript(self, transcript_path):
        utterances = []
        with open(transcript_path, "r") as fp:
            next(fp)  # skip the first line
            for row in fp:
                row_split = row.strip().split(",")
                start_time = float(row_split[0].strip())
                end_time = float(row_split[1].strip())
                transcript = ','.join(row_split[2:-1]).strip()
                confidence = float(row_split[-1].strip())
                length = end_time - start_time

                if not self.check_transcript_valid(confidence, length):
                    # This part of transcript doesn't meet requirements
                    continue

                # Save transcript
                utterance = {'start_time': start_time, 'end_time': end_time, 'transcript': transcript}
                utterances.append(utterance)

        return utterances

    def predict(self, audio_path, transcript_path, process_prediction=None):
        predictions = []
        utterances = self.load_transcript(transcript_path)
        for utterance in utterances:
            audio = self.load_audio(audio_path, utterance['start_time'], utterance['end_time']).unsqueeze(dim=0)
            tokens = self.tokenizer.tokenize(utterance['transcript'])
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(dim=0)
            attention_mask = torch.tensor([1] * len(tokens)).unsqueeze(dim=0)
            prediction = self.predict_utterance(audio, token_ids, attention_mask)
            if process_prediction is not None:
                process_prediction(prediction)
            predictions.append(prediction)
        return predictions

    def predict_utterance(self, audio, token_ids, attention_mask):
        # If there are multiple speakers this will reset the hidden states correctly based on when speakers change
        # but for now, assume single speaker
        participants = ['speaker' for _ in range(audio.size(0))]
        model_inputs = {'audio': audio, 'token_ids': token_ids, 'attention_mask': attention_mask,
                        'participants': participants, 'hx': self.conversational_memory}

        # Make sure no gradients are tracked
        with torch.no_grad():
            predictions, attention, out, self.conversational_memory = self.emotion_recognition_model(**model_inputs)

        emotion_logits, arousal, valence, dominance = predictions

        # Get probabilities
        emotion_probabilities = torch.softmax(emotion_logits[:, self.tuple_of_idx], dim=1)

        # Remove unwanted emotion predictions
        emotion_logits = torch.argmax(emotion_logits[:, self.tuple_of_idx], dim=1)

        emotion_probabilities_str = []
        for probability_vector in emotion_probabilities:
            probabilities = []
            for i, probability in enumerate(probability_vector):
                prob_class = self.tuple_of_idx[i]
                probabilities.append((self.full_names[prob_class], probability))
            emotion_probabilities_str.append(probabilities)

        # This is a list if a batch was submitted
        predicted_emotion = []
        predicted_emotion_str = []
        for logit in emotion_logits:
            prediction_class = self.tuple_of_idx[logit.item()]
            predicted_emotion.append(prediction_class)
            predicted_emotion_str.append(self.full_names[prediction_class])

        arousal_probabilities = torch.softmax(arousal, dim=1)
        valence_probabilities = torch.softmax(arousal, dim=1)
        dominance_probabilities = torch.softmax(arousal, dim=1)
        arousal_probabilities_str = []
        valence_probabilities_str = []
        dominance_probabilities_str = []
        for probability_vector in arousal_probabilities:
            probabilities = []
            for i, probability in enumerate(probability_vector):
                probabilities.append((self.aro_val_dom_names[i], probability))
            arousal_probabilities_str.append(probabilities)
        for probability_vector in valence_probabilities:
            probabilities = []
            for i, probability in enumerate(probability_vector):
                probabilities.append((self.aro_val_dom_names[i], probability))
            valence_probabilities_str.append(probabilities)
        for probability_vector in dominance_probabilities:
            probabilities = []
            for i, probability in enumerate(probability_vector):
                probabilities.append((self.aro_val_dom_names[i], probability))
            dominance_probabilities_str.append(probabilities)

        predicted_arousal = []
        predicted_arousal_str = []
        for logit in torch.argmax(arousal, dim=1):
            prediction_class = logit.item()
            predicted_arousal.append(prediction_class)
            predicted_arousal_str.append(self.aro_val_dom_names[prediction_class])
        predicted_valence = []
        predicted_valence_str = []
        for logit in torch.argmax(valence, dim=1):
            prediction_class = logit.item()
            predicted_valence.append(prediction_class)
            predicted_valence_str.append(self.aro_val_dom_names[prediction_class])
        predicted_dominance = []
        predicted_dominance_str = []
        for logit in torch.argmax(dominance, dim=1):
            prediction_class = logit.item()
            predicted_dominance.append(prediction_class)
            predicted_dominance_str.append(self.aro_val_dom_names[prediction_class])


        details = (predictions, attention, out, arousal, valence, dominance)
        emotion = (predicted_emotion, predicted_emotion_str, emotion_probabilities, emotion_probabilities_str)
        arousal = (predicted_arousal, predicted_arousal_str, arousal_probabilities, arousal_probabilities_str)
        valence = (predicted_valence, predicted_valence_str, valence_probabilities, valence_probabilities_str)
        dominance = (predicted_dominance, predicted_dominance_str, dominance_probabilities, dominance_probabilities_str)
        return emotion, arousal, valence, dominance, details
        # return predicted_emotion, predicted_emotion_str, emotion_probabilities, emotion_probabilities_str, details
