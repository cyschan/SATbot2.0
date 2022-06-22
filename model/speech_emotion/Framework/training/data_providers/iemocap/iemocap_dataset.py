import librosa
import numpy as np
import torch

from torch.utils.data import Dataset
from os import listdir
import re
from model.speech_emotion.Framework.global_config import get_gpu_device


class IEMOCAPDataset(Dataset):
    def __init__(self, root_dir, split='all', emotions=3, input_length=1, mono=False, conversational=False,
                 training_stats_tuple=None):

        """                                     CONFIGURE DATASET PARAMETERS                                         """
        self.root_dir = root_dir
        self.conversational = conversational
        self.full_emotion_list = ['ang', 'hap', 'sad', 'sur', 'neu', 'fru', 'exc', 'fea', 'dis', 'oth']
        self.input_length = input_length
        self.mono = mono
        self.enable_cuda = True
        self.device = get_gpu_device()
        self.regex_to_get_participant = r'(.+)\d\d\d'

        # Choose which emotions to load
        if isinstance(emotions, list):
            # If a list is provided and all elements are valid then use these
            for emotion in emotions:
                if emotion not in self.full_emotion_list:
                    raise ValueError('Invalid emotion detected in requested emotions.')
            self.emotion_list = emotions
            self.number_of_classes = len(emotions)
        elif isinstance(emotions, int):
            # If a number is provided then we use this many emotions by using the first n items in the full emotion list
            self.number_of_classes = emotions
            self.emotion_list = self.full_emotion_list[:emotions]
        else:
            raise ValueError('Emotions should be an integer or list of emotions.')

        # Here we define the different types of splits in the dataset
        self.session_splits = dict()
        self.session_splits['train'] = ['Session1', 'Session2', 'Session3']
        self.session_splits['valid'] = ['Session4']
        self.session_splits['train_valid'] = ['Session1', 'Session2', 'Session3', 'Session4']
        self.session_splits['test'] = ['Session5']
        self.session_splits['all'] = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

        self.split = split
        if self.split not in self.session_splits:
            raise ValueError("Invalid split type.")

        """                                         LOAD DATASET                                                     """
        self.file_stats, self.one_hot_emotion_decoder, self.one_hot_detail_decoder = self.read_stats()
        self.file_paths = self.get_file_paths()
        self.path_to_transcript, self.conversations = self.get_transcripts_and_conversations()

        # Generate stats for normalising the dataset
        if training_stats_tuple is None:
            training_stats_tuple = self.get_training_stats()

        self.training_stats = dict()
        self.training_stats['audio_mean'] = training_stats_tuple[0]
        self.training_stats['audio_std'] = training_stats_tuple[1]

        # Generate some dataset information which can be optionally displayed and get the relative weights of classes
        # in the current split, this can be used for the optimiser during training if desired
        self.dataset_info, self.class_weights = self.get_file_stats(self.file_paths)
        self.number_of_samples = len(self.file_paths)

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, file_path):
        # audio, emotion, arousal, valence, dominance, pre_pad_length = self.get_samples(file_path)
        audio, transcript, emotion, arousal, valence, dominance = self.get_samples(file_path)

        if self.conversational:
            participant = re.match(self.regex_to_get_participant, file_path).group(1)
            return audio, transcript, emotion, arousal, valence, dominance, participant

        return audio, transcript, emotion, arousal, valence, dominance

    def read_stats(self):
        # Create decoders which can optionally be used to decode output vectors from the model
        one_hot_emotion_decoder = dict()
        one_hot_detail_decoder = dict()

        # Need to map each emotion to a position in the one hot encoded vector
        # order should be preserved, e.g. ang -> position 0, hap -> 1, ...
        emotion_position = dict()
        for i, emotion in enumerate(self.emotion_list):
            emotion_position[emotion] = i
            one_hot_emotion_decoder[i] = emotion

        # Similarly one is needed for valence and arousal, which are neg, neu and pos
        aro_val_dom_position = {'neg': 0, 'neu': 1, 'pos': 2}
        one_hot_detail_decoder = {0: 'neg', 1: 'neu', 2: 'pos'}

        stats_dict = dict()
        # Original stats.
        with open(self.root_dir + '/stats.txt', "r") as fp:
            for row in fp:
                row_split = row.strip().split("\t")
                wav_file_name = row_split[0]
                emotion = row_split[1]
                arousal = row_split[2]
                valence = row_split[3]
                dominance = row_split[4]

                if emotion not in self.emotion_list:
                    continue

                if emotion not in emotion_position or arousal not in aro_val_dom_position or \
                        valence not in aro_val_dom_position or dominance not in aro_val_dom_position:
                    raise ValueError("Invalid emotion case.")

                stats_dict[wav_file_name] = dict()

                # One hot encode each vector
                # Emotion
                emotion_vec = [0 for _ in range(self.number_of_classes)]
                emotion_vec[emotion_position[emotion]] = 1
                # Arousal
                arousal_vec = [0 for _ in range(3)]
                arousal_vec[aro_val_dom_position[arousal]] = 1
                # Valence
                valence_vec = [0 for _ in range(3)]
                valence_vec[aro_val_dom_position[valence]] = 1
                # Dominance
                dominance_vec = [0 for _ in range(3)]
                dominance_vec[aro_val_dom_position[dominance]] = 1

                # Add one hot vectors to the stats dictionary
                stats_dict[wav_file_name]["emotion"] = torch.tensor(emotion_vec, dtype=torch.float32)
                stats_dict[wav_file_name]["arousal"] = torch.tensor(arousal_vec, dtype=torch.float32)
                stats_dict[wav_file_name]["valence"] = torch.tensor(valence_vec, dtype=torch.float32)
                stats_dict[wav_file_name]["dominance"] = torch.tensor(dominance_vec, dtype=torch.float32)

        return stats_dict, one_hot_emotion_decoder, one_hot_detail_decoder

    def decode_emotion_vector(self, index):
        # Index is the provided index of the max position in the one hot vector
        if index in self.one_hot_emotion_decoder:
            return self.one_hot_emotion_decoder[index]
        else:
            raise ValueError('Index not valid.')

    def get_file_paths(self):
        list_of_files = librosa.util.find_files(self.root_dir)
        sessions = self.session_splits[self.split]
        file_paths = []
        for file_path in list_of_files:
            if not file_path[-4:] == ".wav":
                raise ValueError("IEMOCAP only has wavs.")

            # Split will be [[...split root dir], "SessionX", "dialog" or "sentences", "wav", segment, wav_file]
            split_file_path = file_path.split("/")
            session_name = split_file_path[-5]
            relative_file_path = "/".join(split_file_path[-5:])

            # Any paths which aren't in the stats file should be skipped
            if relative_file_path not in self.file_stats:
                continue

            if session_name in sessions:
                file_paths.append(relative_file_path)

        return sorted(file_paths)

    def get_transcripts_and_conversations(self):
        single_party_conversations = []
        relative_file_path_to_transcript = dict()
        for session in self.session_splits[self.split]:
            session_transcripts_root_dir = self.root_dir + '/' + session + '/dialog/transcriptions'
            audio_path_head = session + '/sentences/wav'
            # Note - os.listdir returns relative paths
            for file_path in listdir(session_transcripts_root_dir):
                if file_path.startswith('.'):
                    # Skip hidden model_files
                    continue

                if not file_path[-4:] == '.txt':
                    raise ValueError("IEMOCAP transcripts should only be txt model_files.")

                audio_folder_name = file_path[:-4]
                conversation_parties = dict()
                with open(session_transcripts_root_dir + '/' + file_path, "r") as fp:
                    for row in fp:
                        row_split = row.strip().split(' ')
                        audio_file_name = row_split[0]
                        participant = re.match(self.regex_to_get_participant, audio_file_name)
                        if participant:
                            participant = participant.group(1)
                        else:
                            continue
                        # timestamps = row_split[1]
                        transcript = ' '.join(row_split[2:])

                        audio_relative_path = '{}/{}/{}.wav'.format(audio_path_head, audio_folder_name, audio_file_name)
                        if audio_relative_path not in self.file_stats:
                            continue
                        if participant in conversation_parties:
                            conversation_parties[participant].append(audio_relative_path)
                        else:
                            conversation_parties[participant] = [audio_relative_path]
                        relative_file_path_to_transcript[audio_relative_path] = transcript
                single_party_conversations.extend(conversation_parties.values())
        return relative_file_path_to_transcript, single_party_conversations

    # This function calculates the audio mean and audio standard deviation of all audio model_files in the dataset
    def get_training_stats(self):
        audio_sum = 0
        count = 0
        for file_path in self.file_paths:
            audio = librosa.core.load(self.root_dir + "/" + file_path, sr=48000, mono=self.mono)
            audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
            audio_sum += np.sum(audio)
            count += audio.size

        audio_mean = audio_sum / count
        audio_squares_sum = 0
        for file_path in self.file_paths:
            audio = librosa.core.load(self.root_dir + "/" + file_path, sr=48000, mono=self.mono)
            audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
            audio_squares_sum += np.sum(np.power(audio - audio_mean, 2.0))

        audio_std = np.sqrt(audio_squares_sum / count)
        return audio_mean, audio_std

    def get_samples(self, file_path):
        audio = librosa.core.load(self.root_dir + "/" + file_path, sr=48000, mono=self.mono)
        audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)

        # audio = (audio - self.training_stats['audio_mean']) / self.training_stats['audio_std']

        # If GPU is available we want to load the tensors on the GPU
        # Reshape audio to (seq len // input length, input length)
        audio_frames = torch.from_numpy(audio).float().to(self.device).view(-1, self.input_length)
        emotion = self.file_stats[file_path]["emotion"]
        arousal = self.file_stats[file_path]["arousal"]
        valence = self.file_stats[file_path]["valence"]
        dominance = self.file_stats[file_path]["dominance"]
        transcript = self.path_to_transcript[file_path]

        return audio_frames, transcript, emotion, arousal, valence, dominance

    def get_class_weights(self):
        return self.class_weights

    def get_file_stats(self, file_paths):
        stats = dict()
        stats['emotion'] = dict()
        for i in range(len(self.emotion_list)):
            stats['emotion'][i] = []

        stats['arousal'] = dict()
        stats['valence'] = dict()
        stats['dominance'] = dict()
        # 3 for neg, neu, pos.
        for i in range(3):
            stats['arousal'][i] = []
            stats['valence'][i] = []
            stats['dominance'][i] = []

        for file_path in file_paths:
            emotion = torch.argmax(self.file_stats[file_path]["emotion"], 0).item()
            arousal = torch.argmax(self.file_stats[file_path]["arousal"], 0).item()
            valence = torch.argmax(self.file_stats[file_path]["valence"], 0).item()
            dominance = torch.argmax(self.file_stats[file_path]["dominance"], 0).item()
            stats['emotion'][emotion].append(file_path)
            stats['arousal'][arousal].append(file_path)
            stats['valence'][valence].append(file_path)
            stats['dominance'][dominance].append(file_path)

        weight_class = lambda x: 0 if x == 0 else largest_class / x
        self.class_weights = dict()
        for key in stats:
            largest_class = max([len(stats[key][class_num]) for class_num in stats[key]])
            print('largest class in {} - {}'.format(key, largest_class))
            class_weights = [weight_class(len(stats[key][class_num])) for class_num in stats[key]]
            self.class_weights[key] = torch.tensor(class_weights).to(self.device)

        return stats, self.class_weights

    def print_dataset_stats(self):
        print('----- DATASET INFO -----')
        print('-- SPLIT: {} --'.format(self.split))
        print('-- LENGTH: {} --'.format(self.number_of_samples))
        print('-- DATASET BALANCE --')
        for key in self.dataset_info:
            decoder = self.one_hot_emotion_decoder if key == 'emotion' else self.one_hot_detail_decoder
            for class_num in self.dataset_info[key]:
                print('- {} SAMPLES MATCHING {} class {} ({}) -'.format(len(self.dataset_info[key][class_num]), key,
                                                                        class_num, decoder[class_num]))

        print('- EMOTION CLASS WEIGHTS {} -'.format(self.class_weights))