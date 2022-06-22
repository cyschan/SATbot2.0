import math
import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.speech_emotion.Framework.training.data_providers.dataloaders.dataloader import get_iemocap_train_test_split_data_loader
from model.speech_emotion.Framework.utils.model_utils import StatsManager, load_model, save_model
from model.speech_emotion.Framework.utils.validation import validate_parameters
from model.speech_emotion.Framework.global_config import get_gpu_device

from transformers import BertTokenizer


class TrainOnIEMOCAP:
    def __init__(self, training_config, dataset_config, data_loader_config, model):
        print_cuda_stats('Initialising training')
        self.enable_cuda = True
        self.device = get_gpu_device()

        self.training_config = self.setup_training_config(training_config)

        dataloaders, datasets, normalisation_tuple = get_iemocap_train_test_split_data_loader(data_loader_config, dataset_config)

        model.normalisation_tuple = normalisation_tuple
        self.datasets = {'train': datasets[0], 'test': datasets[1]}
        self.dataloaders = {'train': dataloaders[0], 'test': dataloaders[1]}

        for key in self.datasets:
            self.datasets[key].print_dataset_stats()

        self.dataset_lengths = dict()
        self.dataset_lengths['train'] = len(self.datasets['train'])
        self.dataset_lengths['test'] = len(self.datasets['test'])

        print_cuda_stats('datasets loaded')
        print('Train set of length {} using {} to normalise.'.format(self.dataset_lengths['train'],
                                                                     self.datasets['train'].training_stats))
        print('Test set of length {} using {} to normalise.'.format(self.dataset_lengths['test'],
                                                                    self.datasets['test'].training_stats))

        self.stats_manager = StatsManager(self.training_config['model_dir'], self.training_config['stats_file_name'],
                                          model_columns=['emotion_model_file'])

        # Store model
        self.model = model.to(self.device)

        # If model already trained, load this model from the stats.csv
        newest_model_info = self.stats_manager.get_newest_model_path()
        if newest_model_info is not None:
            model_paths, epochs_done, iterations_done = newest_model_info
            if (len(model_paths) > 1):
                raise ValueError('This method only supports training 1 emotion model on IEMOCAP.')

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

        print('Multi Head Model - Parameter ratio alpha (emotion):beta (aro/val/dom) = {}:{}'.format(self.alpha,
                                                                                                     self.beta))

        self.iterations_done = iterations_done
        self.start_epoch = epochs_done
        self.batches_since_update = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'])
        self.model_path = self.training_config['model_dir'] + '/' + self.training_config['model_save_path'] + '_{}.pth'
        class_weights = self.datasets['train'].get_class_weights()
        self.emotion_criterion = self.training_config['criterion_constructor'](weight=class_weights['emotion'])
        self.arousal_criterion = self.training_config['criterion_constructor'](weight=class_weights['arousal'])
        self.valence_criterion = self.training_config['criterion_constructor'](weight=class_weights['valence'])
        self.dominance_criterion = self.training_config['criterion_constructor'](weight=class_weights['dominance'])
        self.loss_lists = dict()
        self.loss_lists['emotion'] = []
        self.loss_lists['arousal'] = []
        self.loss_lists['valence'] = []
        self.loss_lists['dominance'] = []
        self.highest_emotion_acc = None

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
        hx = None

        # Train every batch in dataloader
        for i, batch in enumerate(dataloader):
            # Code often uses too much memory on GPU, old tensors can be sent to cpu after iterations
            (audio, transcript_ids, attn_mask, emotion, arousal, valence, dominance), participants = batch
            del batch

            print_cuda_stats('load batch')
            self.batches_since_update += len(audio)
            self.model.train()

            # Get model outputs and then detach the hidden states to save GPU memory
            predictions, attention, hidden_out, hx = self.model(audio, transcript_ids, attn_mask, participants, hx)
            last_speaker, (hx, cx) = hx
            hx = hx.detach()
            cx = cx.detach()
            hx = (last_speaker, (hx, cx))

            pred_emotions, pred_arousal, pred_valence, pred_dominance = predictions
            del audio, transcript_ids, attn_mask

            # Calculate the loss values
            print('')
            print(pred_emotions.shape, emotion.shape)
            print(pred_emotions, emotion)
            emotion_loss = self.emotion_criterion(pred_emotions, emotion)
            del pred_emotions, emotion
            aro_loss = self.arousal_criterion(pred_arousal, arousal)
            del pred_arousal, arousal
            val_loss = self.valence_criterion(pred_valence, valence)
            del pred_valence, valence
            dom_loss = self.dominance_criterion(pred_dominance, dominance)
            del pred_dominance, dominance
            detail_loss = aro_loss + val_loss + dom_loss

            if torch.cuda.is_available():
                emotion_loss.cuda()
                detail_loss.cuda()

            loss = self.alpha * emotion_loss + self.beta * detail_loss
            loss.backward()

            # If we have done enough samples to have finished batch, update model parameters
            if self.batches_since_update >= self.training_config['update_batch_size']:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.batches_since_update = 0

            self.loss_lists['emotion'].append(self.alpha * emotion_loss.cpu().item())
            self.loss_lists['arousal'].append(self.beta * aro_loss.cpu().item())
            self.loss_lists['valence'].append(self.beta * val_loss.cpu().item())
            self.loss_lists['dominance'].append(self.beta * dom_loss.cpu().item())

            self.iterations_done += 1
            print_cuda_stats('iteration done')
            print('Iteration {}/{} in current training epoch'.format(i, math.ceil(
                datalength / self.training_config['batch_size'])))
            print('Current loss: emotion {} detail {}.'.format(self.alpha * emotion_loss, self.beta * detail_loss))

            torch.cuda.empty_cache()
            print_cuda_stats('cache cleared')

            if self.iterations_done % self.training_config['valid_every_n_steps'] == 0:
                probabilities, labels, true_labels, true_one_hot = self.run_validation_epoch(self.dataloaders['test'],
                                                                                             self.dataset_lengths[
                                                                                                 'test'])

                for key in probabilities:
                    split = 'test_{}'.format(key)
                    pred_probs = probabilities[key]
                    pred_labels = labels[key]
                    true = true_labels[key]
                    true_encoded = true_one_hot[key]
                    test_row = self.stats_manager.add_row([model_path_for_epoch], split, epoch, self.iterations_done,
                                                          pred_probs, pred_labels, true, true_encoded,
                                                          self.loss_lists[key])
                    loss_vals = 'emotion: {} detail: {}'.format(emotion_loss.item(), detail_loss.item())
                    print('{} Iteration: {}. Loss: {}.  T-Accuracy: {}'.format(key, self.iterations_done, loss_vals,
                                                                               test_row['accuracy']))

                    # If this is the emotion stats, check if this is best model thus far and save if so
                    if key == 'emotion' and (
                            self.highest_emotion_acc is None or self.highest_emotion_acc < test_row['accuracy']):
                        print('BEST EMOTION ACCURACY MODEL THUS FAR, SAVING ITERATION #', self.iterations_done)
                        self.highest_emotion_acc = test_row['accuracy']
                        save_model(self.model, self.model_path.format('best_emotion_model'))

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
        logits = dict()
        true_one_hot = dict()

        logits['emotion'] = []
        logits['arousal'] = []
        logits['valence'] = []
        logits['dominance'] = []

        true_one_hot['emotion'] = []
        true_one_hot['arousal'] = []
        true_one_hot['valence'] = []
        true_one_hot['dominance'] = []

        hx = None
        for k, batch in enumerate(dataloader):
            (audio, transcript_ids, attn_mask, emotion, arousal, valence, dominance), participants = batch
            del batch
            print_cuda_stats('validation batch loaded')

            # Store true values
            true_one_hot['emotion'].extend(emotion.detach().cpu())
            true_one_hot['arousal'].extend(arousal.detach().cpu())
            true_one_hot['valence'].extend(valence.detach().cpu())
            true_one_hot['dominance'].extend(dominance.detach().cpu())
            del emotion, arousal, valence, dominance

            print_cuda_stats('deleted true values after storing')

            # Make predictions and track hidden state
            predictions, attention, hidden_out, hx = self.model(audio, transcript_ids, attn_mask, participants, hx)
            last_speaker, (hx, cx) = hx
            hx = hx.detach()
            cx = cx.detach()
            hx = (last_speaker, (hx, cx))

            pred_emotions, pred_arousal, pred_valence, pred_dominance = predictions

            print_cuda_stats('validation predictions made')
            logits['emotion'].extend(pred_emotions.detach().cpu())
            logits['arousal'].extend(pred_arousal.detach().cpu())
            logits['valence'].extend(pred_valence.detach().cpu())
            logits['dominance'].extend(pred_dominance.detach().cpu())

            del pred_emotions, pred_arousal, pred_valence, pred_dominance

            print('Iteration {}/{} in current test epoch'.format(k, math.ceil(
                datalength / self.training_config['batch_size'])))
            print_cuda_stats('validation iteration complete')
            print_cuda_stats('validation tensors deleted')
            torch.cuda.empty_cache()
            print_cuda_stats('validation cuda cache cleared')

        print_cuda_stats('validation epoch complete')
        labels = dict()
        true_labels = dict()
        probabilities = dict()

        for key in logits:
            probabilities[key] = []
            labels[key] = []
            true_labels[key] = []
            for i, logit in enumerate(logits[key]):
                print_cuda_stats('create probabilities/labels post validation epoch')
                probabilities[key].append(F.softmax(logit, dim=0).numpy())
                labels[key].append(torch.max(logit, 0)[1].item())
                true_labels[key].append(torch.max(true_one_hot[key][i], 0)[1].item())
                true_one_hot[key][i] = true_one_hot[key][i].numpy()
                print_cuda_stats('deleting logit')
                # Delete all references to the logit
                logits[key][i] = 0
                del logit
                print_cuda_stats('logit deleted')

        print('validation labels created')
        self.model.train()

        return probabilities, labels, true_labels, true_one_hot

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
