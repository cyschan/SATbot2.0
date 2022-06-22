import math
import os
from os import path

import torch
import torch.nn as nn
import torch.optim as optim

from model.speech_emotion.Framework.training.data_providers.dataloaders.interview_dataloader import \
    get_avec_train_test_split_interview_data_loader
from model.speech_emotion.Framework.utils.model_utils import StatsManager, load_model, save_model
from model.speech_emotion.Framework.utils.validation import validate_parameters
from model.speech_emotion.Framework.global_config import get_gpu_device

from transformers import BertTokenizer


class TrainOnAVEC:
    def __init__(self, training_config, dataset_config, data_loader_config, hidden_state_model, aggregator_model,
                 emotion_model=None):
        print_cuda_stats('Initialising training')
        self.enable_cuda = True
        self.device = get_gpu_device()

        self.training_config = self.setup_training_config(training_config)

        dataloaders, datasets, normalisation_tuple = get_avec_train_test_split_interview_data_loader(data_loader_config,
                                                                                                     dataset_config)

        hidden_state_model.normalisation_tuple = normalisation_tuple

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
                                          model_columns=['hidden_state_model_file', 'aggregator_model_file'],
                                          stat_columns=stats)

        # Load models
        # In this case one model creates hidden states for each utterance while a different model aggregates all the
        # hidden states down to a final prediction
        self.hidden_state_model = hidden_state_model.to(self.device)
        self.aggregator_model = aggregator_model.to(self.device)

        # Assume the emotion model is setup and has the best model already loaded, set to evaluation mode and freeze
        if emotion_model is not None:
            self.emotion_model = emotion_model.to(self.device).eval()
            for param in self.emotion_model.parameters():
                param.requires_grad = False
        else:
            self.emotion_model = None

        # If model already trained, load this model from the stats.csv
        newest_model_info = self.stats_manager.get_newest_model_path()
        if newest_model_info is not None:
            model_paths, epochs_done, iterations_done = newest_model_info

            hs_expected_name = '{}_hidden_state_{}.pth'.format(self.training_config['model_save_path'], epochs_done)
            hs_model_path = model_paths[0]
            if hs_expected_name != hs_model_path.split('/')[-1]:
                print('WARNING - Name of the saved model path and the model do not match, ensure this is correct.')
            agg_expected_name = '{}_aggregator_{}.pth'.format(self.training_config['model_save_path'], epochs_done)
            agg_model_path = model_paths[1]
            if agg_expected_name != agg_model_path.split('/')[-1]:
                print('WARNING - Name of the saved model path and the model do not match, ensure this is correct.')

            print('Loading models...')

            load_model(self.hidden_state_model, hs_model_path)
            print('Successfully loaded model {}.'.format(hs_model_path))
            load_model(self.aggregator_model, agg_model_path)
            print('Successfully loaded model {}.'.format(agg_model_path))
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
        self.conversations_since_update = 0
        self.hidden_state_optimizer = optim.Adam(self.hidden_state_model.parameters(),
                                                 lr=self.training_config['learning_rate'])
        self.aggregator_optimizer = optim.Adam(self.aggregator_model.parameters(),
                                               lr=self.training_config['learning_rate'])
        self.hs_model_path = self.training_config['model_dir'] + '/' + self.training_config[
            'model_save_path'] + '_hidden_state_{}.pth'
        self.agg_model_path = self.training_config['model_dir'] + '/' + self.training_config[
            'model_save_path'] + '_aggregator_{}.pth'
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
        hs_model_path_for_epoch = self.hs_model_path.format(epoch)
        agg_model_path_for_epoch = self.agg_model_path.format(epoch)

        # Train every batch in dataloader
        for i, batch in enumerate(dataloader):
            # The batch is a tuple containing dataloaders (loads each utterance in the interview), labels as tensors,
            # and a list of lists, which contain the speaker of each utterance, but each list should all be the same
            # speaker since it is split by interview
            conversation_data_loaders, conversation_labels, participants_batch = batch

            # The labels are gender, phq_score, pcl_c_score, ptsd_severity_score
            gender, phq_8, pcl_c, ptsd_severity = conversation_labels
            phq_losses = []
            ptsd_losses = []

            for j, utterance_loader in enumerate(conversation_data_loaders):
                phq_label = phq_8[j, :].clone()
                ptsd_label = ptsd_severity[j, :].clone()
                hidden_states, emotion_logits = self.run_iteration(utterance_loader, participants_batch[j], phq_label, ptsd_label)

                predictions, utterance_predictions, utterance_weights = self.aggregator_model(hidden_states, emotion_logits)
                phq_8_pred, ptsd_sev_pred = predictions

                # Calculate the loss values
                phq_loss = self.phq_criterion(phq_8_pred, phq_label).to(self.device)
                ptsd_loss = self.ptsd_criterion(ptsd_sev_pred, ptsd_label).to(self.device)

                # Update loss per conversation, not per batch
                loss = self.alpha * phq_loss + self.beta * ptsd_loss
                loss.backward()

                phq_losses.append(self.alpha * phq_loss.cpu().item())
                ptsd_losses.append(self.beta * ptsd_loss.cpu().item())
                self.loss_lists['phq_8'].append(self.alpha * phq_loss.cpu().item())
                self.loss_lists['ptsd_severity'].append(self.beta * ptsd_loss.cpu().item())

                self.conversations_since_update += 1

                # If we have done enough samples to have finished batch, update model parameters
                if self.conversations_since_update >= self.training_config['update_batch_size']:
                    self.hidden_state_optimizer.step()
                    self.aggregator_optimizer.step()
                    self.hidden_state_optimizer.zero_grad()
                    self.aggregator_optimizer.zero_grad()
                    self.conversations_since_update = 0

            self.iterations_done += 1
            print_cuda_stats('iteration done')
            print('Iteration {}/{} in current training epoch'.format(i, math.ceil(
                datalength / self.training_config['batch_size'])))
            print('Average loss for iteration: PHQ 8 {} PTSD Severity {}.'.format(sum(phq_losses) / len(phq_losses),
                                                                                  sum(ptsd_losses) / len(ptsd_losses)))

            torch.cuda.empty_cache()
            print_cuda_stats('cache cleared')

            if self.iterations_done % self.training_config['valid_every_n_steps'] == 0:
                predictions, true_values = self.run_validation_epoch(self.dataloaders['test'],
                                                                     self.dataset_lengths['test'])

                for key in predictions:
                    split = 'test_{}'.format(key)
                    prediction = predictions[key]
                    true_value = true_values[key]
                    test_row = self.stats_manager.add_avec_row([hs_model_path_for_epoch, agg_model_path_for_epoch],
                                                               split, epoch, self.iterations_done, prediction,
                                                               true_value, self.loss_lists[key])
                    avg_phq_loss = sum(self.loss_lists['phq_8']) / len(self.loss_lists['phq_8'])
                    avg_ptsd_loss = sum(self.loss_lists['ptsd_severity']) / len(self.loss_lists['ptsd_severity'])
                    loss_vals = 'PHQ 8: {} PTSD Severity: {}'.format(avg_phq_loss, avg_ptsd_loss)
                    print('{} Iteration: {}. Average Loss: {}.  RMSE: {}'.format(key, self.iterations_done, loss_vals,
                                                                                 test_row['rmse']))

                    # If this is the emotion stats, check if this is best model thus far and save if so
                    if key == 'phq_8' and (self.lowest_phq_rmse is None or self.lowest_phq_rmse > test_row['rmse']):
                        print('BEST PHQ 8 RMSE MODEL THUS FAR, SAVING ITERATION #', self.iterations_done)
                        self.lowest_phq_rmse = test_row['rmse']
                        save_model(self.hidden_state_model, self.hs_model_path.format('best_phq_8_model'))
                        save_model(self.aggregator_model, self.agg_model_path.format('best_phq_8_model'))

                self.stats_manager.save_stats()

                save_model(self.hidden_state_model, hs_model_path_for_epoch)
                save_model(self.aggregator_model, agg_model_path_for_epoch)

                if self.training_config['delete_old_models']:
                    old_model = self.hs_model_path.format(epoch - 1)
                    if path.exists(old_model):
                        os.remove(old_model)
                    old_model = self.agg_model_path.format(epoch - 1)
                    if path.exists(old_model):
                        os.remove(old_model)

                for key in self.loss_lists:
                    self.loss_lists[key] = []

    def run_validation_epoch(self, dataloader, datalength):
        self.hidden_state_model.eval()
        self.aggregator_model.eval()

        print_cuda_stats('running validation')
        predictions = dict()
        true_values = dict()

        predictions['phq_8'] = []
        predictions['ptsd_severity'] = []

        true_values['phq_8'] = []
        true_values['ptsd_severity'] = []

        for i, batch in enumerate(dataloader):
            # The batch is a tuple containing dataloaders (loads each utterance in the interview), labels as tensors,
            # and a list of lists, which contain the speaker of each utterance, but each list should all be the same
            # speaker since it is split by interview
            conversation_data_loaders, conversation_labels, participants_batch = batch

            # The labels are gender, phq_score, pcl_c_score, ptsd_severity_score
            gender, phq_8, pcl_c, ptsd_severity = conversation_labels

            true_values['phq_8'].extend(phq_8.detach().cpu())
            true_values['ptsd_severity'].extend(ptsd_severity.detach().cpu())

            phq_8_predictions = []
            ptsd_severity_predictions = []

            for j, utterance_loader in enumerate(conversation_data_loaders):
                phq_label = phq_8[j, :]
                ptsd_label = ptsd_severity[j, :]
                hidden_states, emotion_logits = self.run_iteration(utterance_loader, participants_batch[j], phq_label, ptsd_label)

                preds, utterance_predictions, utterance_weights = self.aggregator_model(hidden_states, emotion_logits)
                phq_8_pred, ptsd_sev_pred = preds
                phq_8_predictions.append(phq_8_pred)
                ptsd_severity_predictions.append(ptsd_sev_pred)

            phq_8_predictions = torch.cat(phq_8_predictions).view(-1, 1)
            ptsd_severity_predictions = torch.cat(ptsd_severity_predictions).view(-1, 1)

            print_cuda_stats('validation predictions made')
            predictions['phq_8'].extend(phq_8_predictions.detach().cpu())
            predictions['ptsd_severity'].extend(ptsd_severity_predictions.detach().cpu())

            del phq_8_predictions, ptsd_severity_predictions

            print('Iteration {}/{} in current test epoch'.format(i, math.ceil(
                datalength / self.training_config['batch_size'])))
            print_cuda_stats('validation iteration complete')
            print_cuda_stats('validation tensors deleted')
            torch.cuda.empty_cache()
            print_cuda_stats('validation cuda cache cleared')

        print_cuda_stats('validation epoch complete')
        self.hidden_state_model.train()
        self.aggregator_model.train()

        return predictions, true_values

    def run_iteration(self, utterance_loader, participants, phq_label, ptsd_label):
        training = self.hidden_state_model.training and self.aggregator_model.training

        if training:
            print_cuda_stats('Training predictions per utterance')
            self.train_sub_groups(utterance_loader, participants, (phq_label, ptsd_label))

        print_cuda_stats('Make prediction for interview')
        # First get the predictions and logits
        hidden_states, emotion_logits = self.get_hidden_states(utterance_loader, participants)

        # We now have 2 lists, with the hidden state and possibly emotion logits for each utterance in the interview
        hidden_states = torch.cat(hidden_states)

        if self.emotion_model:
            emotion_logits = torch.cat(emotion_logits)

        return hidden_states, emotion_logits

    def train_sub_groups(self, utterance_loader, participants, labels, sub_batch_size=8):
        # We want to train on smaller groups of utterances to avoid memory issues, we can train both models in tandem
        # here
        hidden_states = []
        emotion_logits = [] if self.emotion_model else None
        hx_hidden_state = None
        hx_emotion = None
        batch_start_idx = 0
        batch_end_idx = 0
        training = self.hidden_state_model.training and self.aggregator_model.training
        update_required = False
        for j, sample in enumerate(utterance_loader):
            update_required = True
            print_cuda_stats('Training values for utterance {}'.format(j))
            # This is a batch of utterances
            audio, transcript_ids, attn_mask, transcript_confidence = sample
            batch_size = audio.size(0)
            batch_end_idx += batch_size
            batch_participants = participants[batch_start_idx:batch_end_idx]
            batch_start_idx = batch_end_idx
            failed = True
            while failed:
                try:
                    hidden_state, attention, out, hx_hidden_state = self.hidden_state_model(audio, transcript_ids, attn_mask,
                                                                                            batch_participants, hx_hidden_state)
                    failed = False
                except RuntimeError:
                    failed = True
                    print('Failed to process hidden state')
            hx_hidden_state = self.detach_hx(hx_hidden_state)

            if not training:
                hidden_state = hidden_state.detach()
                hx_hidden_state = self.detach_hx(hx_hidden_state)

            hidden_states.append(hidden_state)

            if self.emotion_model:
                predictions, attention, out, hx_emotion = self.emotion_model(audio, transcript_ids, attn_mask,
                                                                             batch_participants, hx_emotion)
                hx_emotion = self.detach_hx(hx_emotion)

                # Flatten logits into an emotional logit vector
                # flat_logits = torch.cat(predictions, dim=1)
                emotion_logits.append(out)

            # We want to update every ~4 utterances, so every 2 sub batches
            if (j + 1) % 2 == 0 and training:
                print_cuda_stats('Back propagating previous utterances')
                hidden_states = torch.cat(hidden_states)
                if hidden_states[-1].requires_grad: hidden_states.retain_grad()
                # print(hidden_states)

                if self.emotion_model:
                    emotion_logits = torch.cat(emotion_logits)

                predictions, utterance_predictions, utterance_weights = self.aggregator_model(hidden_states,
                                                                                              emotion_logits)

                self.back_propagate(predictions, labels, 0.25)
                print(hidden_states.shape)
                print('{} Utterances trained in current interview'.format(hidden_states.size(0)))

                hx_hidden_state = self.detach_hx(hx_hidden_state)
                hidden_states = hidden_states.detach()
                hidden_states = [hidden_states]

                if self.emotion_model:
                    emotion_logits = emotion_logits.detach()
                    emotion_logits = [emotion_logits]

                update_required = False

            if (j + 1) % sub_batch_size == 0:
                # Update the parameters
                print('Updating weights on sub batch')
                self.hidden_state_optimizer.step()
                self.aggregator_optimizer.step()
                self.hidden_state_optimizer.zero_grad()
                self.aggregator_optimizer.zero_grad()

        if update_required:
            # Loss hasn't been back propagated for the last few states
            print_cuda_stats('Back propagating previous utterances')
            hidden_states = torch.cat(hidden_states)
            if hidden_states[-1].requires_grad: hidden_states.retain_grad()

            if self.emotion_model:
                emotion_logits = torch.cat(emotion_logits)

            predictions, utterance_predictions, utterance_weights = self.aggregator_model(hidden_states,
                                                                                          emotion_logits)

            self.back_propagate(predictions, labels, 0.25)
            print(hidden_states.shape)
            print('{} Utterances trained in current interview'.format(hidden_states.size(0)))

        print('Updating weights after finishing sub group')
        self.hidden_state_optimizer.step()
        self.aggregator_optimizer.step()
        self.hidden_state_optimizer.zero_grad()
        self.aggregator_optimizer.zero_grad()

    def get_hidden_states(self, utterance_loader, participants):
        # We don't want to have gradients here to avoid any issues with memory, this will be used to train only the
        # aggregator function
        self.hidden_state_model.eval()
        with torch.no_grad():
            hidden_states = []
            emotion_logits = [] if self.emotion_model else None
            hx_hidden_state = None
            hx_emotion = None
            batch_start_idx = 0
            batch_end_idx = 0
            for j, sample in enumerate(utterance_loader):
                print_cuda_stats('Generating values for utterance {}'.format(j))
                # This is a batch of utterances
                audio, transcript_ids, attn_mask, transcript_confidence = sample
                batch_size = audio.size(0)
                batch_end_idx += batch_size
                batch_participants = participants[batch_start_idx:batch_end_idx]
                batch_start_idx = batch_end_idx
                hidden_state, attention, out, hx_hidden_state = self.hidden_state_model(audio, transcript_ids,
                                                                                        attn_mask,
                                                                                        batch_participants,
                                                                                        hx_hidden_state)
                hidden_states.append(hidden_state)

                if self.emotion_model:
                    predictions, attention, out, hx_emotion = self.emotion_model(audio, transcript_ids, attn_mask,
                                                                                 batch_participants, hx_emotion)
                    # Flatten logits into an emotional logit vector
                    # flat_logits = torch.cat(predictions, dim=1)
                    emotion_logits.append(out)

        self.hidden_state_model.train()
        return hidden_states, emotion_logits

    def back_propagate(self, predictions, labels, loss_weight=1.0):
        phq_8_prediction, ptsd_prediction = predictions
        phq_label, ptsd_label = labels
        phq_loss = self.phq_criterion(phq_8_prediction, phq_label).to(self.device)
        ptsd_loss = self.ptsd_criterion(ptsd_prediction, ptsd_label).to(self.device)

        # Lower value of loss since this is a midway prediction
        loss = loss_weight * (self.alpha * phq_loss + self.beta * ptsd_loss)
        print('hidden_state_model training:', self.hidden_state_model.training)
        print('aggregator_model training:', self.aggregator_model.training)
        emotion_model_out = 'No Model' if self.emotion_model is None else self.emotion_model.training
        print('emotion_model training:', emotion_model_out)
        loss.backward()

        print_cuda_stats('Utterance group done')
        print('Utterance group loss: PHQ 8 {} PTSD Severity {}.'.format(self.alpha * phq_loss,
                                                                        self.beta * ptsd_loss))

    def detach_hx(self, hx):
        # Don't want to back propogate all the way through time to the first iteration when hx was created
        last_speaker, (hx, cx) = hx
        hx = hx.detach()
        cx = cx.detach()
        hx = (last_speaker, (hx, cx))
        return hx

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
