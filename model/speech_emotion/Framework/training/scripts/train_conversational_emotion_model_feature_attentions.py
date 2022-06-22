import torch.nn as nn

from model.speech_emotion.Framework.models.emotion_recognition.emotion_recognition_model_full_att import EmotionRecognitionModel
from model.speech_emotion.Framework.training.training_loops.iemocap_training_loop import TrainOnIEMOCAP

# Create config for the training loop
training_config = dict()
training_config['model_dir'] = '/vol/bitbucket/jat415/checkpoints/final_stretch/emotion'
training_config['num_epochs'] = 10

# Want to use a batch size of 32 but memory issues limits to 4. So use a batch size of 4 but only apply update every
# update_batch_size / 4 batches
training_config['batch_size'] = 2
training_config['update_batch_size'] = 32

training_config['valid_every_n_steps'] = 1086
# dict_of_training_params['criterion_constructor'] = nn.BCEWithLogitsLoss
training_config['criterion_constructor'] = nn.CrossEntropyLoss
training_config['learning_rate'] = 0.00001
training_config['model_save_path'] = 'emotion_model_shared_1024_attention'
training_config['stats_file_name'] = 'emotion_model_shared_1024_attention'
training_config['alpha'] = 3

# Create config for the dataset
dataset_config = dict()
dataset_config['root_dir'] = '/vol/bitbucket/jat415/data/IEMOCAP_audio_only'
dataset_config['split'] = 'train_valid'

# Calculation of these stats is done if they aren't provided but it is extremely time consuming > 1 hour on gpucluster
# these are the training stats for the train_valid split
dataset_config['training_stats_tuple'] = (-8.541886477750582e-06, 0.05738852859500303)

# Only 2 examples of disgust in the entire dataset and so it has to be ignored
# dataset_config['emotions'] = ['ang', 'hap', 'sad', 'sur', 'neu', 'fru', 'exc', 'fea', 'dis']
dataset_config['emotions'] = ['ang', 'hap', 'sad', 'sur', 'neu', 'fru', 'exc', 'fea']
dataset_config['conversational'] = True

# Create config for the dataloader
data_loader_config = dict()
data_loader_config['batch_size'] = training_config['batch_size']
data_loader_config['shuffle_train_set'] = True
data_loader_config['conversational'] = True
data_loader_config['one_hot_encode_train_targets'] = False
data_loader_config['one_hot_encode_test_targets'] = True

# Create model
model = EmotionRecognitionModel(input_size=1, hidden_size=1024, number_of_emotions=len(dataset_config['emotions']),
                                num_layers_bilstm=2, cell_state_clipping=None, bias=True, freeze_bert=False,
                                predict_emotion=True, predict_details=True)

trainer = TrainOnIEMOCAP(training_config, dataset_config, data_loader_config, model)

trainer.start_training()
