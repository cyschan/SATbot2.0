from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model.speech_emotion.Framework.training.data_providers.dataloaders.sequence_padder import PadSequences, ConversationSampler
from model.speech_emotion.Framework.training.data_providers.avec.avec_utterence_dataset import AVECDataset
from model.speech_emotion.Framework.training.data_providers.iemocap.iemocap_dataset import IEMOCAPDataset
from model.speech_emotion.Framework.utils.validation import validate_parameters


def get_data_loader(dataset, batch_size, shuffle, pad_both_sides=False, sequence_length_dimension=0,
                    one_hot_target=True, bert_tokenizer=BertTokenizer, pre_trained_bert='bert-large-uncased'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=PadSequences(sequence_length_dimension, pad_both_sides, one_hot_target,
                                                bert_tokenizer, pre_trained_bert))
    return loader


def get_conversation_data_loader(dataset, batch_size, shuffle, pad_both_sides=False, sequence_length_dimension=0,
                                 one_hot_target=True, bert_tokenizer=BertTokenizer,
                                 pre_trained_bert='bert-large-uncased'):
    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=ConversationSampler(dataset, shuffle),
                        collate_fn=PadSequences(sequence_length_dimension, pad_both_sides, one_hot_target,
                                                bert_tokenizer, pre_trained_bert))
    return loader


def get_iemocap_train_test_split_data_loader(data_loader_config, dataset_config):
    # Define the arguments required for the data loader objects
    required_data_loader_arguments = ['batch_size', 'shuffle_train_set']
    default_data_loader_arguments = [('pad_both_sides', False), ('sequence_length_dimension', 0),
                                     ('one_hot_encode_train_targets', True), ('one_hot_encode_test_targets', True),
                                     ('bert_tokenizer', BertTokenizer), ('pre_trained_bert', 'bert-large-uncased'),
                                     ('conversational', False), ('float_labels', False)]

    # Define the arguments required for loading the actual datasets
    required_dataset_arguments = ['root_dir']
    default_dataset_arguments = [('train_split', 'train'), ('emotions', 3), ('input_length', 1), ('mono', False),
                                 ('conversational', False), ('training_stats_tuple', None)]

    # Validate the input config dictionaries
    data_loader_config = validate_parameters(required_data_loader_arguments, default_data_loader_arguments,
                                             data_loader_config)
    dataset_config = validate_parameters(required_dataset_arguments, default_dataset_arguments, dataset_config)

    # Validate that the data split is okay, train uses train set for training, train_valid uses train + valid instead
    if dataset_config['train_split'] != 'train' and dataset_config['train_split'] != 'train_valid':
        raise ValueError('Train split must be either "train" set or "train_valid" set')

    training_stats_tuple = None
    if 'training_stats_tuple' in dataset_config:
        training_stats_tuple = dataset_config['training_stats_tuple']

    # Create train dataset
    train_set = IEMOCAPDataset(dataset_config['root_dir'], dataset_config['train_split'], dataset_config['emotions'],
                               dataset_config['input_length'], dataset_config['mono'], dataset_config['conversational'],
                               training_stats_tuple)

    # Get the training stats tuple from the test set and apply this to the train set
    training_stats_tuple = (train_set.training_stats['audio_mean'], train_set.training_stats['audio_std'])

    # Create test dataset
    test_set = IEMOCAPDataset(dataset_config['root_dir'], 'test', dataset_config['emotions'],
                              dataset_config['input_length'], dataset_config['mono'], dataset_config['conversational'],
                              training_stats_tuple)

    # Now create the dataloaders
    data_loader_config['one_hot_target'] = data_loader_config['one_hot_encode_train_targets']
    train_loader = build_data_loader(data_loader_config, train_set)
    data_loader_config['one_hot_target'] = data_loader_config['one_hot_encode_test_targets']
    test_loader = build_data_loader(data_loader_config, test_set)

    return (train_loader, test_loader), (train_set, test_set), training_stats_tuple


def get_avec_train_test_split_data_loader(data_loader_config, dataset_config):
    # Define the arguments required for the data loader objects
    required_data_loader_arguments = ['batch_size', 'shuffle_train_set']
    default_data_loader_arguments = [('pad_both_sides', False), ('sequence_length_dimension', 0),
                                     ('one_hot_target', True), ('bert_tokenizer', BertTokenizer),
                                     ('pre_trained_bert', 'bert-large-uncased'), ('conversational', False),
                                     ('float_labels', False)]

    # Define the arguments required for loading the actual datasets
    required_dataset_arguments = ['root_dir', 'min_audio_length', 'max_audio_length', 'min_confidence']
    default_dataset_arguments = [('train_split', 'train'), ('input_length', 1), ('mono', False),
                                 ('conversational', False), ('training_stats_tuple', None)]

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

    print('TRAINING STATS', training_stats_tuple)
    # Create train dataset
    train_set = AVECDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                            dataset_config['max_audio_length'], dataset_config['min_confidence'],
                            dataset_config['train_split'], dataset_config['input_length'], dataset_config['mono'],
                            dataset_config['conversational'], training_stats_tuple)

    # Get the training stats tuple from the test set and apply this to the train set
    training_stats_tuple = (train_set.training_stats['audio_mean'], train_set.training_stats['audio_std'])

    # Create test dataset
    test_set = AVECDataset(dataset_config['root_dir'], dataset_config['min_audio_length'],
                           dataset_config['max_audio_length'], dataset_config['min_confidence'],
                           dataset_config['train_split'], dataset_config['input_length'], dataset_config['mono'],
                           dataset_config['conversational'], training_stats_tuple)

    # Now create the dataloaders
    train_loader = build_data_loader(data_loader_config, train_set)
    test_loader = build_data_loader(data_loader_config, test_set)

    return (train_loader, test_loader), (train_set, test_set)


def build_data_loader(data_loader_config, dataset):
    # Sampler only needed for conversational training
    data_loader_sampler = None
    if data_loader_config['conversational']:
        data_loader_sampler = ConversationSampler(dataset, data_loader_config['shuffle_train_set'])

    sequence_padder = PadSequences(data_loader_config['sequence_length_dimension'],
                                   data_loader_config['pad_both_sides'], data_loader_config['one_hot_target'],
                                   data_loader_config['bert_tokenizer'], data_loader_config['pre_trained_bert'],
                                   data_loader_config['conversational'], data_loader_config['float_labels'])

    data_loader = DataLoader(dataset, batch_size=data_loader_config['batch_size'], sampler=data_loader_sampler,
                             collate_fn=sequence_padder)

    return data_loader
