import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler
from transformers import BertTokenizer

import random

from model.speech_emotion.Framework.training.data_providers.covid_interviews.covid_interview_dataset import CovidInterviewDataset
from model.speech_emotion.Framework.global_config import get_gpu_device
from model.speech_emotion.Framework.training.data_providers.avec.avec_interview_dataset import AVECInterviewDataset
from model.speech_emotion.Framework.training.data_providers.dataloaders.dataloader import PadSequences
from model.speech_emotion.Framework.utils.validation import validate_parameters


class collate_interviews:
    def __init__(self):
        self.enable_cuda = True
        self.device = get_gpu_device()

    def __call__(self, batch):
        # Batch will be a list of (dataloader, labels, participants)
        batch_size = len(batch)
        dataloaders = []
        tensor_labels = []
        participants = []
        # Create tensor label objects with correct shape, all labels should be same shape so get it from the first
        for label in batch[0][1]:
            label_length = len(label) if list(label.shape) else 1
            tensor_labels.append(torch.zeros((batch_size, label_length)).float().to(self.device))

        for i, sample in enumerate(batch):
            # For this batch convert each label to tensor
            dataloader, labels, participant = sample
            for j, label in enumerate(labels):
                tensor_labels[j][i, :] = label

            dataloaders.append(dataloader)
            participants.append(participant)

        return dataloaders, tensor_labels, participants


def get_avec_train_test_split_interview_data_loader(data_loader_config, dataset_config, test_set_str='test'):
    # Define the arguments required for the data loader objects
    required_data_loader_arguments = ['batch_size', 'sub_batch_size', 'shuffle_train_set']
    default_data_loader_arguments = [('pad_both_sides', False), ('sequence_length_dimension', 0),
                                     ('one_hot_target', True), ('bert_tokenizer', BertTokenizer),
                                     ('pre_trained_bert', 'bert-large-uncased'), ('conversational', False),
                                     ('float_labels', False)]

    # Define the arguments required for loading the actual datasets
    required_dataset_arguments = ['root_dir', 'min_audio_length', 'max_audio_length', 'min_confidence']
    default_dataset_arguments = [('train_split', 'train'), ('input_length', 1), ('mono', False),
                                 ('conversational', False), ('training_stats_tuple', None), ('per_utterance', False)]

    # Validate the input config dictionaries
    data_loader_config = validate_parameters(required_data_loader_arguments, default_data_loader_arguments,
                                             data_loader_config)
    dataset_config = validate_parameters(required_dataset_arguments, default_dataset_arguments, dataset_config)

    # Validate that the data split is okay, train uses train set for training, train_valid uses train + valid instead
    # TODO: add train_vald - and dataset_config['train_split'] != 'train_valid':
    if dataset_config['train_split'] != 'train':
        raise ValueError('Train split must be either "train" set (TODO: add "train_valid" set)')

    training_stats_tuple = None
    if 'training_stats_tuple' in dataset_config:
        training_stats_tuple = dataset_config['training_stats_tuple']

    # Define the data loader we want to use on the individual conversations
    data_loader_constructor = lambda conversation: build_data_loader(data_loader_config, conversation)

    # Create train dataset
    train_set = AVECInterviewDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                                     dataset_config['max_audio_length'], dataset_config['min_confidence'],
                                     dataset_config['train_split'], dataset_config['input_length'],
                                     dataset_config['mono'], data_loader_constructor, training_stats_tuple)

    print('TRAINING STATS', training_stats_tuple)

    # Get the training stats tuple from the test set and apply this to the train set
    training_stats_tuple = (train_set.training_stats['audio_mean'], train_set.training_stats['audio_std'])

    # Create test dataset
    test_set = AVECInterviewDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                                    dataset_config['max_audio_length'], dataset_config['min_confidence'],
                                    test_set_str, dataset_config['input_length'], dataset_config['mono'],
                                    data_loader_constructor, training_stats_tuple)

    # Now create the dataloaders
    train_loader = DataLoader(train_set, batch_size=data_loader_config['batch_size'], collate_fn=collate_interviews())
    test_loader = DataLoader(test_set, batch_size=data_loader_config['batch_size'], collate_fn=collate_interviews())

    return (train_loader, test_loader), (train_set, test_set), training_stats_tuple


def get_covid_train_test_split_interview_data_loader(data_loader_config, dataset_config):
    # Define the arguments required for the data loader objects
    required_data_loader_arguments = ['batch_size', 'sub_batch_size', 'shuffle_train_set']
    default_data_loader_arguments = [('pad_both_sides', False), ('sequence_length_dimension', 0),
                                     ('one_hot_target', True), ('bert_tokenizer', BertTokenizer),
                                     ('pre_trained_bert', 'bert-large-uncased'), ('conversational', False),
                                     ('float_labels', False)]

    # Define the arguments required for loading the actual datasets
    required_dataset_arguments = ['root_dir', 'min_audio_length', 'max_audio_length', 'min_confidence']
    default_dataset_arguments = [('train_split', 'train'), ('input_length', 1), ('mono', False),
                                 ('conversational', False), ('training_stats_tuple', None), ('per_utterance', False)]

    # Validate the input config dictionaries
    data_loader_config = validate_parameters(required_data_loader_arguments, default_data_loader_arguments,
                                             data_loader_config)
    dataset_config = validate_parameters(required_dataset_arguments, default_dataset_arguments, dataset_config)

    # Validate that the data split is okay, train uses train set for training, train_valid uses train + valid instead
    # TODO: add train_vald - and dataset_config['train_split'] != 'train_valid':
    if dataset_config['train_split'] != 'train':
        raise ValueError('Train split must be either "train" set (TODO: add "train_valid" set)')

    training_stats_tuple = None
    if 'training_stats_tuple' in dataset_config:
        training_stats_tuple = dataset_config['training_stats_tuple']

    # Define the data loader we want to use on the individual conversations
    data_loader_constructor = lambda conversation: build_data_loader(data_loader_config, conversation)

    # Create train dataset
    train_set = CovidInterviewDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                                      dataset_config['max_audio_length'], dataset_config['min_confidence'],
                                      dataset_config['train_split'], dataset_config['input_length'],
                                      dataset_config['mono'], data_loader_constructor, training_stats_tuple)

    print('TRAINING STATS', training_stats_tuple)

    # Get the training stats tuple from the test set and apply this to the train set
    training_stats_tuple = (train_set.training_stats['audio_mean'], train_set.training_stats['audio_std'])

    # Create test dataset
    test_set = CovidInterviewDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                                     dataset_config['max_audio_length'], dataset_config['min_confidence'],
                                     'test', dataset_config['input_length'], dataset_config['mono'],
                                     data_loader_constructor, training_stats_tuple)

    # Now create the dataloaders
    train_loader = DataLoader(train_set, batch_size=data_loader_config['batch_size'], collate_fn=collate_interviews())
    test_loader = DataLoader(test_set, batch_size=data_loader_config['batch_size'], collate_fn=collate_interviews())

    return (train_loader, test_loader), (train_set, test_set), training_stats_tuple


def build_data_loader(data_loader_config, dataset):
    # Sampler only needed for conversational training
    sequence_padder = PadSequences(data_loader_config['sequence_length_dimension'],
                                   data_loader_config['pad_both_sides'], data_loader_config['one_hot_target'],
                                   data_loader_config['bert_tokenizer'], data_loader_config['pre_trained_bert'],
                                   data_loader_config['conversational'], data_loader_config['float_labels'])

    data_loader = DataLoader(dataset, batch_size=data_loader_config['sub_batch_size'], collate_fn=sequence_padder)

    return data_loader
