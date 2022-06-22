import random

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import BertTokenizer
from model.speech_emotion.Framework.global_config import get_gpu_device


class PadSequences:
    def __init__(self, dimension=0, pad_both_sides=False, one_hot_target=True, bert_tokenizer=BertTokenizer,
                 pre_trained_bert='bert-large-uncased', conversational=False, float_labels=False):
        self.enable_cuda = True
        self.device = get_gpu_device()
        self.dimension = dimension  # This is which dimension to pad
        self.pad_both_sides = pad_both_sides
        self.expected_sizes = None
        self.one_hot_target = one_hot_target
        self.bert_tokenizer = bert_tokenizer.from_pretrained(pre_trained_bert)
        self.conversational = conversational
        self.float_labels = float_labels
        # If we are loading conversations the final dimension represents the speakers and is not a label
        self.last_label_index = -1 if conversational else None

    def pad_sample(self, sample, max_seq_len, max_words, skipped_dims):
        # Split the tensors and the tokens which represent words used for the current BERT tokenizer
        tensor, tokens = sample
        original_size = tensor.size(self.dimension)
        padding_to_add = max_seq_len - original_size

        # F.pad expects pairs of ints representing left padding and right padding, the dimensions are also done starting
        # from -1, -2, -3, ... So for each skipped dimension we add 2 zeroes
        padding = [0] * (skipped_dims * 2)

        # Now add a pair representing the padding we need to add to the current sample
        left_pad = padding_to_add // 2 if self.pad_both_sides else 0
        right_pad = padding_to_add - left_pad
        padding.append(left_pad)
        padding.append(right_pad)

        padded_tensor = F.pad(tensor, padding)

        pad_tokens_to_add = max_words - len(tokens)
        padded_tokens = tokens + ['[PAD]'] * pad_tokens_to_add
        padded_ids = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(padded_tokens))
        attention_mask = torch.tensor([1] * len(tokens) + [0] * pad_tokens_to_add)
        return padded_tensor, padded_ids.long(), attention_mask

    def calculate_sizes(self, batch):
        batch_size = len(batch)
        sample = batch[0]
        audio_shape = sample[0].shape
        input_size = audio_shape[-1]

        # Padding starts at the final dimension so we need to skip padding on dimensions which come after the desired
        # dimension
        # e.g. to pad dimension 2 (3rd dimension) in 3d tensor, we don't skip any
        #      to pad dimension 0 (1st dimension) in 4d tensor, we skip 3
        number_of_dimensions = len(audio_shape)
        dimensions_to_skip = number_of_dimensions - self.dimension - 1

        # Calculate the lengths for the labels, these will all be PyTorch one-hot-encoded tensors, we want the shapes
        # with batch length as the first dimension
        label_shapes = []
        for label in sample[2:self.last_label_index]:
            length = len(label) if list(label.shape) else 1
            label_shapes.append((batch_size, length))

        self.expected_sizes = (batch_size, input_size, dimensions_to_skip, label_shapes)

    def tokenize_transcript(self, transcript):
        tokens = self.bert_tokenizer.tokenize(transcript)
        return ['[CLS]'] + tokens + ['[SEP]']

    def __call__(self, batch):
        # Each value in batch should be [audio, transcript, labels...]
        # where audio = tensor of audio, transcript = transcript of audio
        # labels will be lists of e.g. one hot encoded categories

        if self.expected_sizes is None:
            self.calculate_sizes(batch)

        batch_size, input_size, dimensions_to_skip, label_shapes = self.expected_sizes
        if batch_size != len(batch):
            # Incomplete batch loaded i.e. length of dataset not divisible by batch size
            batch_size = len(batch)
            label_shapes = [(batch_size, label_shape[1]) for label_shape in label_shapes]

        seq_lens = [value[0].size(self.dimension) for value in batch]
        max_seq_len = max(seq_lens)

        tokenized_transcripts = [self.tokenize_transcript(sample[1]) for sample in batch]
        text_lens = [len(tokens) for tokens in tokenized_transcripts]
        max_words = max(text_lens)

        # Converts the inputs into list of tensors with padded audio, will be [audio, transcript, labels...]
        tensors = [torch.zeros((batch_size, max_seq_len, input_size)).to(self.device),
                   torch.zeros((batch_size, max_words)).long().to(self.device),
                   torch.zeros((batch_size, max_words)).to(self.device)]

        for label_shape in label_shapes:
            if self.float_labels:
                tensors.append(torch.zeros(label_shape).float().to(self.device))
            else:
                tensors.append(torch.zeros(label_shape).long().to(self.device))

        for i, sample in enumerate(batch):
            sample_to_pad = (sample[0], tokenized_transcripts[i])
            aud, ids, attn = self.pad_sample(sample_to_pad, max_seq_len, max_words, dimensions_to_skip)
            tensors[0][i, :, :] = aud
            tensors[1][i, :] = ids
            tensors[2][i, :] = attn
            for j, label in enumerate(sample[2:self.last_label_index]):
                tensors[3 + j][i, :] = label

        if not self.one_hot_target:
            for j in range(len(tensors) - 3):
                tensors[3 + j] = torch.argmax(tensors[3 + j], 1)

        # If conversational make the return value into a tuple to return (tensors, list of ordered speakers in batch)
        if self.conversational:
            tensors = (tensors, [sample[-1] for sample in batch])

        return tensors


class ConversationSampler(Sampler):
    def __init__(self, data, shuffle):
        super().__init__(data)
        self.conversations = data.conversations
        if shuffle:
            random.shuffle(self.conversations)

        self.conversations = [file_path for conversation in self.conversations for file_path in conversation]

    def __iter__(self):
        return iter(self.conversations)

    def __len__(self):
        len(self.conversations)
