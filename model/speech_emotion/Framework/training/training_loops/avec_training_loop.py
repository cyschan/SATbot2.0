import math
import os
from os import path

import torch
import torch.nn as nn
import torch.optim as optim

from model.speech_emotion.Framework.training.data_providers.dataloaders.dataloader import get_avec_train_test_split_data_loader
from model.speech_emotion.Framework.utils.model_utils import StatsManager, load_model, save_model
from model.speech_emotion.Framework.utils.validation import validate_parameters
from model.speech_emotion.Framework.global_config import get_gpu_device

from transformers import BertTokenizer


class TrainOnAVEC:
    def __init__(self, training_config, dataset_config, data_loader_config, model, emotion_model=None):
        print_cuda_stats('Initialising training')
        self.enable_cuda = True
        self.device = get_gpu_device()

        self.training_config = self.setup_training_config(training_config)

        dataloaders, datasets = get_avec_train_test_split_data_loader(data_loader_config, dataset_config)
        self.datasets = {'train': datasets[0], 'test': datasets[1]}
        self.dataloaders = {'train': dataloaders[0], 'test': dataloaders[1]}

        self.dataset_lengths = dict()
        self.dataset_lengths['train'] = len(self.datasets['train'])
        self.dataset_lengths['test'] = len(self.datasets['test'])

        print_cuda_stats('datasets loaded')
        print('Train set of length {} using {} to normalise.'.format(self.dataset_lengths['train'],
                                                                     self.datasets['train'].training_stats))
        print('Test set of length {} using {} to normalise.'.format(self.dataset_lengths['test'],
                                                                    self.datasets['test'].training_stats))

        # Stats file was built with training_loops in mind, override the columns saved in the csv to match avec
        stats = ['data_split', 'epochs_trained', 'iterations_trained', 'rmse', 'mae', 'mse']
        self.stats_manager = StatsManager(self.training_config['model_dir'], self.training_config['stats_file_name'],
                                          model_columns=['emotion_model_file'], stat_columns=stats)

        # Load models
        self.model = model.to(self.device)

        # Assume the emotion model is setup and has the best model already loaded, set to evaluation mode and freeze
        self.emotion_model = emotion_model.to(self.device).eval()
        if self.emotion_model is not None:
            for param in self.emotion_model.parameters():
                param.requires_grad = False

        # If model already trained, load this model from the stats.csv
        newest_model_info = self.stats_manager.get_newest_model_path()
        if newest_model_info is not None:
            model_paths, epochs_done, iterations_done = newest_model_info
            if (len(model_paths) > 1):
                raise ValueError('This method only supports training 1 model on AVEC.')

            expected_name = '{}_{}.pth'.format(self.training_config['model_save_path'], epochs_done)
            model_path = model_paths[0]
            if expected_name != model_path.split('/')[-1]:
                print('WARNING - Name of the saved model path and the model do not match, ensure this is correct.')

            print('Loading models...')

            load_model(self.model, model_path)
            print('Successfully loaded model {}.'.format(model_path))
            print('Loaded model already trained for {} epochs ({} iterations)'.format(epochs_done, iterations_done))
        else:
            iterations_done = 0
            epochs_done = 0

        # alpha parameter holds the ratio i.e. alpha = 2 => alpha = 2, self.beta = 1, alpha = 0.5 => alpha = 1, self.beta = 2
        self.alpha = self.training_config['alpha']
        if self.alpha < 0:
            raise ValueError('Value of alpha not between 0 and 1.')
        if self.alpha < 1:
            self.beta = 1 / self.alpha
            self.alpha = 1
        else:
            self.beta = 1

        print('Multi Head Model - Parameter ratio alpha (PHQ 8):beta (PTSD) = {}:{}'.format(self.alpha, self.beta))

        self.iterations_done = iterations_done
        self.start_epoch = epochs_done
        self.batches_since_update = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'])
        self.model_path = self.training_config['model_dir'] + '/' + self.training_config['model_save_path'] + '_{}.pth'
        self.phq_criterion = self.training_config['criterion_constructor']()
        self.ptsd_criterion = self.training_config['criterion_constructor']()
        self.loss_lists = dict()
        self.loss_lists['phq_8'] = []
        self.loss_lists['ptsd_severity'] = []
        self.lowest_phq_rmse = None

        print_cuda_stats('models loaded')

    def start_training(self):
        print_cuda_stats('start training')
        for epoch in range(self.start_epoch, self.start_epoch + self.training_config['num_epochs']):
            print('epoch {}/{}...'.format(epoch, self.training_config['num_epochs'] + self.start_epoch))
            self.run_training_epoch(self.dataloaders['train'], self.dataset_lengths['train'], epoch)

    def run_training_epoch(self, dataloader, datalength, epoch):
        # Model path should be a string containing "{}" which we can substitute for current epoch number
        model_path_for_epoch = self.model_path.format(epoch)

        # Initialise hx
        hx_emotion = None
        hx_phq = None

        # Train every batch in dataloader
        for i, batch in enumerate(dataloader):
            # Code often uses too much memory on GPU, old tensors can be sent to cpu after iterations
            sample, participants = batch
            audio, transcript_ids, attn_mask, transcript_confidence, gender, phq_8, pcl_c, ptsd_severity = sample
            del batch, sample

            print_cuda_stats('load batch')
            self.batches_since_update += len(audio)
            self.model.train()

            # Get model outputs and then detach the hidden states to save GPU memory
            if self.emotion_model is not None:
                emotion_predictions, attention, hidden_out, hx_emotion = self.emotion_model(audio, transcript_ids,
                                                                                            attn_mask, participants,
                                                                                            hx_emotion)
                last_speaker, (hx_emotion, cx) = hx_emotion
                hx_emotion = hx_emotion.detach()
                cx = cx.detach()
                hx_emotion = (last_speaker, (hx_emotion, cx))

                avec_predictions, attention, hidden_out, hx_phq = self.model(audio, transcript_ids, attn_mask,
                                                                             emotion_predictions, participants, hx_phq)

                del emotion_predictions
            else:
                avec_predictions, attention, hidden_out, hx_phq = self.model(audio, transcript_ids, attn_mask,
                                                                             participants, hx_phq)

            last_speaker, (hx_phq, cx) = hx_phq
            hx_phq = hx_phq.detach()
            cx = cx.detach()
            hx_phq = (last_speaker, (hx_phq, cx))

            del audio, transcript_ids, attn_mask, hidden_out, attention

            pred_phq_8, pred_ptsd_severity = avec_predictions

            # Calculate the loss values
            print('')
            print(pred_phq_8.shape, phq_8.shape)
            print(pred_phq_8, phq_8)
            phq_loss = self.phq_criterion(pred_phq_8, phq_8)
            del pred_phq_8, phq_8
            ptsd_loss = self.ptsd_criterion(pred_ptsd_severity, ptsd_severity)
            del pred_ptsd_severity, ptsd_severity

            if torch.cuda.is_available():
                phq_loss.cuda()
                ptsd_loss.cuda()

            loss = self.alpha * phq_loss + self.beta * ptsd_loss
            loss.backward()

            # If we have done enough samples to have finished batch, update model parameters
            if self.batches_since_update >= self.training_config['update_batch_size']:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.batches_since_update = 0

            self.loss_lists['phq_8'].append(self.alpha * phq_loss.cpu().item())
            self.loss_lists['ptsd_severity'].append(self.beta * ptsd_loss.cpu().item())

            self.iterations_done += 1
            print_cuda_stats('iteration done')
            print('Iteration {}/{} in current training epoch'.format(i, math.ceil(
                datalength / self.training_config['batch_size'])))
            print('Current loss: PHQ 8 {} PTSD Severity {}.'.format(self.alpha * phq_loss, self.beta * ptsd_loss))

            torch.cuda.empty_cache()
            print_cuda_stats('cache cleared')

            if self.iterations_done % self.training_config['valid_every_n_steps'] == 0:
                predictions, true_values = self.run_validation_epoch(self.dataloaders['test'],
                                                                     self.dataset_lengths['test'])

                for key in predictions:
                    split = 'test_{}'.format(key)
                    prediction = predictions[key]
                    true_value = true_values[key]
                    test_row = self.stats_manager.add_avec_row([model_path_for_epoch], split, epoch, self.iterations_done,
                                                               prediction, true_value, self.loss_lists[key])
                    loss_vals = 'PHQ 8: {} PTSD Severity: {}'.format(phq_loss.item(), ptsd_loss.item())
                    print('{} Iteration: {}. Loss: {}.  RMSE: {}'.format(key, self.iterations_done, loss_vals,
                                                                         test_row['rmse']))

                    # If this is the emotion stats, check if this is best model thus far and save if so
                    if key == 'phq_8' and (self.lowest_phq_rmse is None or self.lowest_phq_rmse > test_row['rmse']):
                        print('BEST PHQ 8 RMSE MODEL THUS FAR, SAVING ITERATION #', self.iterations_done)
                        self.lowest_phq_rmse = test_row['rmse']
                        save_model(self.model, self.model_path.format('best_phq_8_model'))

                self.stats_manager.save_stats()

                save_model(self.model, model_path_for_epoch)

                if self.training_config['delete_old_models']:
                    old_model = self.model_path.format(epoch - 1)
                    if path.exists(old_model):
                        os.remove(old_model)

                for key in self.loss_lists:
                    self.loss_lists[key] = []

    def run_validation_epoch(self, dataloader, datalength):
        self.model.eval()

        print_cuda_stats('running validation')
        predictions = dict()
        true_values = dict()

        predictions['phq_8'] = []
        predictions['ptsd_severity'] = []

        true_values['phq_8'] = []
        true_values['ptsd_severity'] = []

        hx_emotion = None
        hx_phq = None
        for k, batch in enumerate(dataloader):
            sample, participants = batch
            audio, transcript_ids, attn_mask, transcript_confidence, gender, phq_8, pcl_c, ptsd_severity = sample
            del batch, sample

            print_cuda_stats('validation batch loaded')

            # Store true values
            true_values['phq_8'].extend(phq_8.detach().cpu())
            true_values['ptsd_severity'].extend(ptsd_severity.detach().cpu())
            del transcript_confidence, gender, phq_8, pcl_c, ptsd_severity

            print_cuda_stats('deleted true values after storing')

            # Make predictions and track hidden state
            if self.emotion_model is not None:
                emotion_predictions, attention, hidden_out, hx_emotion = self.emotion_model(audio, transcript_ids,
                                                                                            attn_mask, participants,
                                                                                            hx_emotion)
                last_speaker, (hx_emotion, cx) = hx_emotion
                hx_emotion = hx_emotion.detach()
                cx = cx.detach()
                hx_emotion = (last_speaker, (hx_emotion, cx))

                avec_predictions, attention, hidden_out, hx_phq = self.model(audio, transcript_ids, attn_mask,
                                                                             emotion_predictions, participants, hx_phq)
                del emotion_predictions
            else:
                avec_predictions, attention, hidden_out, hx_phq = self.model(audio, transcript_ids, attn_mask,
                                                                             participants, hx_phq)
            last_speaker, (hx_phq, cx) = hx_phq
            hx_phq = hx_phq.detach()
            cx = cx.detach()
            hx_phq = (last_speaker, (hx_phq, cx))

            del audio, transcript_ids, attn_mask, hidden_out, attention

            pred_phq_8, pred_ptsd_severity = avec_predictions

            print_cuda_stats('validation predictions made')
            predictions['phq_8'].extend(pred_phq_8.detach().cpu())
            predictions['ptsd_severity'].extend(pred_ptsd_severity.detach().cpu())

            del avec_predictions, pred_phq_8, pred_ptsd_severity

            print('Iteration {}/{} in current test epoch'.format(k, math.ceil(
                datalength / self.training_config['batch_size'])))
            print_cuda_stats('validation iteration complete')
            print_cuda_stats('validation tensors deleted')
            torch.cuda.empty_cache()
            print_cuda_stats('validation cuda cache cleared')

        print_cuda_stats('validation epoch complete')

        return predictions, true_values

    def setup_training_config(self, training_config):
        required_arguments = ['model_dir', 'num_epochs', 'batch_size', 'valid_every_n_steps', 'model_save_path',
                              'update_batch_size']
        default_arguments = [('learning_rate', 0.00001), ('criterion_constructor', nn.CrossEntropyLoss),
                             ('stats_file_name', 'stats'), ('alpha', 0.5), ('bert_tokenizer', BertTokenizer),
                             ('pre_trained_bert', 'bert-large-uncased'), ('delete_old_models', True)]

        return validate_parameters(required_arguments, default_arguments, training_config)


disable_prints = False


def print_cuda_stats(msg):
    if disable_prints:
        return
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))
