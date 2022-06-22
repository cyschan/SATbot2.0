import torch.nn as nn

from model.speech_emotion.Framework.models.phq_prediction.phq_hidden_state_model import HiddenStateModel
from model.speech_emotion.Framework.models.phq_prediction.phq_aggregator_model import AggregatorModel
from model.speech_emotion.Framework.training.training_loops.avec_interview_training_loop import TrainOnAVEC

# Create config for the training loop
training_config = dict()
training_config['model_dir'] = '/vol/bitbucket/jat415/checkpoints/final_stretch/phq'
training_config['num_epochs'] = 2

# Want to use a batch size of 32 but memory issues limits to 4. So use a batch size of 4 but only apply update every
# update_batch_size / 4 batches
training_config['batch_size'] = 2
training_config['update_batch_size'] = 4

training_config['valid_every_n_steps'] = 10
# dict_of_training_params['criterion_constructor'] = nn.BCEWithLogitsLoss
training_config['criterion_constructor'] = nn.MSELoss
training_config['learning_rate'] = 0.00001
training_config['model_save_path'] = 'depression_prediction_baseline'
training_config['stats_file_name'] = 'depression_prediction_baseline'
training_config['alpha'] = 3

# Create config for the dataset
dataset_config = dict()
dataset_config['root_dir'] = '/vol/bitbucket/jat415/data/AVEC_Challenge_Packed'
dataset_config['split'] = 'train'
dataset_config['conversational'] = True
dataset_config['min_audio_length'] = 1.0
dataset_config['max_audio_length'] = 10
dataset_config['min_confidence'] = 0.85
dataset_config['per_utterance'] = False

# Calculation of these stats is done if they aren't provided but it is extremely time consuming > 1 hour on gpucluster
# these are the training stats for the train_valid split
dataset_config['training_stats_tuple'] = (-2.1862424047454064e-05, 0.016074780249712255)

# Create config for the dataloader
data_loader_config = dict()
data_loader_config['batch_size'] = training_config['batch_size']
data_loader_config['sub_batch_size'] = 2
data_loader_config['shuffle_train_set'] = True
data_loader_config['conversational'] = False
data_loader_config['one_hot_encode_train_targets'] = False
data_loader_config['one_hot_encode_test_targets'] = False
data_loader_config['float_labels'] = True

for run in range(5):
    print('Training model #{}'.format(run))
    training_config['model_save_path'] = 'depression_prediction_baseline_run_{}'.format(run)
    training_config['stats_file_name'] = 'depression_prediction_baseline_run_{}'.format(run)

    # Create models
    emotion_model = None
    
    # Create hidden state model
    hs_model = HiddenStateModel(input_size=1, hidden_size=1024, num_layers_bilstm=2, cell_state_clipping=None, bias=True,
                                freeze_bert=False, predict_phq_8=True, predict_ptsd_severity=True)
    
    # Create aggregator model
    agg_model = AggregatorModel(phq_hidden_size=1024, phq_shrink_hidden_size=1024)
    
    # emotion_logit_length is 17 since we predict 8 logits for emotion and then 3 each for arousal/valence/dominance
    trainer = TrainOnAVEC(training_config, dataset_config, data_loader_config, hs_model, agg_model, emotion_model)
    
    trainer.start_training()
