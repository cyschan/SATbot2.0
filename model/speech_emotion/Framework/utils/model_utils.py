# Calculate the padding size to perform the "same" padding from tensorflow
import math
import copy
import sklearn
import torch
import pandas as pd


from os import path


def same_padding_1d(input_size, kernel_size, stride):
    # Output size = floor((size - k + 2p)/stride) + 1
    # So to maintain original size p = ceiling((size*(stride - 1) - stride + k)/2)
    twice_padding = input_size * (stride - 1) - stride + kernel_size
    padding = math.ceil(twice_padding / 2)
    if (kernel_size % 2 == 0):
        padding = (padding, padding + 1)
    # print(padding)
    return padding



def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))


class StatsManager:
    def __init__(self, model_dir, filename='stats', model_columns=None, stat_columns=None):

        if model_columns is None:
            model_columns = ['model_file']

        if stat_columns is None:
            stat_columns = ['data_split', 'epochs_trained', 'iterations_trained', 'accuracy', 'ham_loss', 'macro_f1',
                                        'au_roc_macro', 'au_roc_micro', 'au_prc_macro', 'au_prc_micro', 'losses']

        self.stats_file = model_dir + '/' + filename + '.csv'
        self.model_columns = model_columns

        # Should calculate accuracy, macro f1, roc_auc_score, average_precision_score
        self.columns = model_columns + stat_columns
        if path.exists(self.stats_file):
            self.df = pd.read_csv(self.stats_file)
        else:
            # data_split is test or validation splits the stats are from
            self.df = pd.DataFrame(columns=self.columns)

    def add_row(self, model_file_paths, split, epoch, iteration, predicted_probs, predicted_labels, true_labels,
                true_one_hot, losses=None):

        if losses is None:
            losses = []

        new_row = dict()

        for i, col_name in enumerate(self.model_columns):
            new_row[col_name] = model_file_paths[i]

        new_row['data_split'] = split
        new_row['epochs_trained'] = epoch
        new_row['iterations_trained'] = iteration

        # Accuracy
        # expects 1d array of labels for true and predicted
        try:
            new_row['accuracy'] = sklearn.metrics.accuracy_score(true_labels, predicted_labels, normalize=True, sample_weight=None)
        except ValueError:
            pass

        # Hamming Loss
        try:
            new_row['ham_loss'] = sklearn.metrics.hamming_loss(true_labels, predicted_labels, sample_weight=None)
        except ValueError:
            pass

        # Macro-F1
        # expects 1d array of labels for true and predicted
        try:
            new_row['macro_f1'] = sklearn.metrics.f1_score(true_labels, predicted_labels, average="macro")
        except ValueError:
            pass
        # AU-ROC.
        # since multiclass case expects true labels to be 1d array (n_samples,), predicted should be probability vector
        try:
            new_row['au_roc_macro'] = sklearn.metrics.roc_auc_score(true_one_hot, predicted_probs, average="macro")
        except ValueError:
            pass
        try:
            new_row['au_roc_micro'] = sklearn.metrics.roc_auc_score(true_one_hot, predicted_probs, average="micro")
        except ValueError:
            pass
        # AU-PRC
        # multi class isn't compatible here so we have to use multi label, so we need to convert the
        try:
            new_row['au_prc_macro'] = sklearn.metrics.average_precision_score(true_one_hot, predicted_probs, average="macro")
        except ValueError:
            pass
        try:
            new_row['au_prc_micro'] = sklearn.metrics.average_precision_score(true_one_hot, predicted_probs, average="micro")
        except ValueError:
            pass

        new_row['losses'] = ' '.join([str(loss) for loss in losses])
        self.df = self.df.append(new_row, ignore_index=True)

        return new_row

    def add_avec_row(self, model_file_paths, split, epoch, iteration, prediction, true_value, losses=None):
        if losses is None:
            losses = []

        new_row = dict()

        for i, col_name in enumerate(self.model_columns):
            new_row[col_name] = model_file_paths[i]

        new_row['data_split'] = split
        new_row['epochs_trained'] = epoch
        new_row['iterations_trained'] = iteration

        # Root Mean Squared Error
        new_row['rmse'] = sklearn.metrics.mean_squared_error(true_value, prediction, squared=False)

        # Mean Absolute Error
        new_row['mae'] = sklearn.metrics.mean_absolute_error(true_value, prediction)

        # Mean Squared Error
        new_row['mse'] = sklearn.metrics.mean_squared_error(true_value, prediction, squared=True)

        new_row['losses'] = ' '.join([str(loss) for loss in losses])
        self.df = self.df.append(new_row, ignore_index=True)

        return new_row

    def save_manual_row(self, dict):
        self.df = self.df.append(dict, ignore_index=True)

    def save_list_as_row(self, row_list):
        if (len(row_list) != len(self.columns)):
            raise ValueError('Length of new row ({}) and number of columns ({}) not equal.'.format(len(row_list),
                                                                                                   len(self.columns)))
        new_row = dict()
        for i, col_name in enumerate(self.columns):
            new_row[col_name] = row_list[i]

        self.df = self.df.append(new_row, ignore_index=True)

        return new_row

    def save_stats(self):
        self.df.to_csv(self.stats_file, index=False)

    def get_newest_model_path(self):

        if self.df.empty:
            return None

        newest_model_idx = self.df['epochs_trained'].argmax()
        row = self.df.iloc[newest_model_idx]

        file_paths = []
        for col_name in self.model_columns:
            file_paths.append(row[col_name])

        return file_paths, row['epochs_trained'], row['iterations_trained']
