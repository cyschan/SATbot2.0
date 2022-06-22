import librosa
import numpy as np
import torch
import re

from torch.utils.data import Dataset, IterableDataset, DataLoader
from os import listdir
from model.speech_emotion.Framework.global_config import get_gpu_device


# Each sample in this dataset loads entire interviews as the individual sample by returning a batch of dataloaders
# these dataloaders will load, in the correct order, each utterance made by the speaker in the interview
class AVECInterviewDataset(Dataset):
    def __init__(self, root_dir, min_audio_length, max_audio_length, min_confidence, split='all', input_length=1,
                 mono=False, data_loader_constructor=DataLoader, training_stats_tuple=None):

        """                                     CONFIGURE DATASET PARAMETERS                                         """
        self.root_dir = root_dir
        self.labels = ['PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
        self.input_length = input_length
        self.mono = mono
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.min_confidence = min_confidence
        self.data_loader_constructor = data_loader_constructor
        self.enable_cuda = True
        self.device = get_gpu_device()
        self.regex_to_get_participant = r'^(\d\d\d)'

        # Here we define the different types of splits in the dataset
        self.split_to_stats = dict()
        self.split_to_stats['train'] = [self.root_dir + '/labels/train_split.csv']
        self.split_to_stats['valid'] = [self.root_dir + '/labels/dev_split.csv']
        self.split_to_stats['test'] = [self.root_dir + '/labels/test_split.csv']
        self.split_to_stats['all'] = [self.root_dir + '/labels/train_split.csv',
                                      self.root_dir + '/labels/dev_split.csv',
                                      self.root_dir + '/labels/test_split.csv']

        self.split = split
        if self.split not in self.split_to_stats:
            raise ValueError("Invalid split type.")

        """                                         LOAD DATASET                                                     """
        # Stats will be PHQ scores, PTSD scores, and transcripts.
        # The tuples will be (audio file path, id, start time, end time)
        self.id_to_stats, self.audio_tuples, self.conversations = self.read_stats()

        # Generate stats for normalising the dataset
        if training_stats_tuple is None:
            training_stats_tuple = self.get_training_stats()

        self.training_stats = dict()
        self.training_stats['audio_mean'] = training_stats_tuple[0]
        self.training_stats['audio_std'] = training_stats_tuple[1]

        self.number_of_samples = len(self.conversations)

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        # Want to return a full interview, so get the conversation for the index and return iterator
        # Speaker list for each utterance, this should all be the same speaker
        conversation = self.conversations[idx]
        speaker = [audio_tuple[0] for audio_tuple in conversation]
        print('getting data for interview', speaker[0])

        # Now create conversation dataset so that it can be used as part of a dataloader
        conversation_dataset = AVECConversation(conversation, self.get_data)

        # Now create the data loader defined by the constructor given to the dataset
        conversation_loader = self.data_loader_constructor(conversation_dataset)

        # Since all these are from the same conversation we only need to get the labels for the conversation once
        # but first assert that all speakers are the same, just in case
        assert all(x == speaker[0] for x in speaker)
        labels = self.get_labels(speaker[0])

        return conversation_loader, labels, speaker

    def read_stats(self):
        # Need to encode gender in case it is used for learning
        gender_dict = {'male': 0, 'female': 1}

        stats_dict = dict()

        # Now for each of the training splits
        for stats_file in self.split_to_stats[self.split]:
            # Original stats.
            with open(stats_file, "r") as fp:
                next(fp)  # skip the first line
                for row in fp:
                    row_split = row.strip().split(",")
                    participant_id = int(row_split[0].strip())
                    gender = row_split[1].strip()
                    PHQ_Score = row_split[3].strip()
                    PCL_C_PTSD = row_split[4].strip()
                    PTSD_Severity = row_split[5].strip()

                    # There are 2 ptsd scores which are NaN so these need to be skipped
                    if PCL_C_PTSD == 'NaN' or PTSD_Severity == 'NaN':
                        continue

                    if participant_id not in range(300, 718):
                        continue

                    stats_dict[participant_id] = dict()

                    # Add one hot vectors to the stats dictionary
                    stats_dict[participant_id]["Gender"] = torch.tensor([gender_dict[gender]], dtype=torch.float32).to(
                        self.device)
                    stats_dict[participant_id]["PHQ Total Score"] = torch.tensor([int(PHQ_Score)],
                                                                                 dtype=torch.float32).to(self.device)
                    stats_dict[participant_id]["PCL C (PTSD) Score"] = torch.tensor([int(PCL_C_PTSD)],
                                                                                    dtype=torch.float32).to(self.device)
                    stats_dict[participant_id]["PTSD Severity Score"] = torch.tensor([int(PTSD_Severity)],
                                                                                     dtype=torch.float32).to(
                        self.device)
                    stats_dict[participant_id]["Transcripts"] = dict()
                    stats_dict[participant_id]['Transcript Confidence'] = dict()

        # Now get the transcripts for each id
        audio_tuples = dict()
        conversations = []
        for participant_id in stats_dict:
            # Single participant conversation so just track each item
            conversation = []
            audio_path = self.root_dir + '/data/{}_P/{}_AUDIO.wav'.format(participant_id, participant_id)
            transcript_path = self.root_dir + '/data/{}_P/{}_Transcript.csv'.format(participant_id, participant_id)
            # TODO: Load the video features
            with open(transcript_path, "r") as fp:
                next(fp)  # skip the first line
                for row in fp:
                    row_split = row.strip().split(",")
                    start_time = float(row_split[0].strip())
                    end_time = float(row_split[1].strip())
                    transcript = ','.join(row_split[2:-1]).strip()
                    confidence = float(row_split[-1].strip())
                    length = end_time - start_time

                    if confidence < self.min_confidence or length < self.min_audio_length \
                            or length > self.max_audio_length:
                        continue

                    # Save transcript
                    time_string = '{}-{}'.format(start_time, end_time)
                    stats_dict[participant_id]["Transcripts"][time_string] = transcript

                    # Save confidence
                    stats_dict[participant_id]['Transcript Confidence'][time_string] = torch.tensor([confidence],
                                                                                                    dtype=torch.float32).to(
                        self.device)

                    # Save audio tuple
                    audio_tuple = (participant_id, audio_path, start_time, end_time, length, time_string)
                    conversation_string = '{}_{}_{}_{}_{}_{}'.format(*audio_tuple)
                    audio_tuples[conversation_string] = audio_tuple
                    conversation.append(audio_tuple)
            # Sort conversation ordered by starting time
            conversations.append(sorted(conversation, key=lambda audio_tuple: audio_tuple[2]))

        return stats_dict, audio_tuples, conversations

    def get_data(self, audio_tuple):
        # Audio tuple is (id, audio path, start time, end time, length, time string)
        participant_id, audio_path, start, end, length, time_str = audio_tuple
        audio = librosa.core.load(audio_path, sr=48000, mono=self.mono, offset=start, duration=length)
        audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)

        # audio = (audio - self.training_stats['audio_mean']) / self.training_stats['audio_std']
        # Move normalisation to model

        audio_frames = torch.from_numpy(audio).float().to(self.device).reshape(-1, self.input_length)
        stats = self.id_to_stats[participant_id]
        transcript = stats['Transcripts'][time_str]
        transcript_confidence = stats['Transcript Confidence'][time_str]

        return audio_frames, transcript, transcript_confidence

    def get_labels(self, participant_id):
        stats = self.id_to_stats[participant_id]
        gender = stats['Gender']
        phq_score = stats['PHQ Total Score']
        pcl_c_score = stats['PCL C (PTSD) Score']
        ptsd_severity_score = stats['PTSD Severity Score']
        return gender, phq_score, pcl_c_score, ptsd_severity_score

    def get_training_stats(self):
        audio_sum = 0
        count = 0
        # Audio tuple is (id, audio path, start time, end time, length, time string)
        for audio_tuple_key in self.audio_tuples:
            audio_tuple = self.audio_tuples[audio_tuple_key]
            _, file_path, start_time, _, length, _ = audio_tuple
            audio = librosa.core.load(file_path, sr=48000, mono=self.mono, offset=start_time, duration=length)
            audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
            audio_sum += np.sum(audio)
            count += audio.size

        audio_mean = audio_sum / count
        audio_squares_sum = 0
        for audio_tuple_key in self.audio_tuples:
            audio_tuple = self.audio_tuples[audio_tuple_key]
            _, file_path, start_time, _, length, _ = audio_tuple
            audio = librosa.core.load(file_path, sr=48000, mono=self.mono, offset=start_time, duration=length)
            audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000).astype(np.float32)
            audio_squares_sum += np.sum(np.power(audio - audio_mean, 2.0))

        audio_std = np.sqrt(audio_squares_sum / count)
        return audio_mean, audio_std

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

class AVECConversation(IterableDataset):
    def __init__(self, conversation, get_function):
        # Conversation should be a list of the audio tuples that can be iterated to get the tensors
        self.conversation = conversation
        self.length = len(conversation)

        # get_function is the method for getting the tensor values from the dataset
        self.get_function = get_function

        # Current position
        self.idx = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration

        values = self.get_function(self.conversation[self.idx])
        self.idx += 1
        return values
