import pytorch_lightning as pl
import textdistance as td
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoModelWithLMHead,
    AutoTokenizer
)
from tokenizers import ByteLevelBPETokenizer

from tokenizers.processors import BertProcessing

#T5:
class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

args_dict = dict(
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
)

args = argparse.Namespace(**args_dict)


#load emotion classifier (T5 small)
with torch.no_grad():
    emo_model = T5FineTuner(args)
    emo_model.load_state_dict(torch.load('T5_small_emotion.pt', map_location=torch.device('cpu')))

#load empathy classifier (T5 small)
with torch.no_grad():
    emp_model = T5FineTuner(args)
    emp_model.load_state_dict(torch.load('T5_small_empathy.pt', map_location=torch.device('cpu'))) #change path

#Load pre-trained GPT2 language model weights
with torch.no_grad():
    gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    gptmodel.eval()

#Load pre-trained GPT2 tokenizer
gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()


def get_emotion(text):
  '''
  Computes and returns an emotion label given an utterance
  '''
  text = re.sub(r'[^\w\s]', '', text)
  text = text.lower()
  with torch.no_grad():
      input_ids = emo_model.tokenizer.encode(text + '</s>', return_tensors='pt')
      output = emo_model.model.generate(input_ids=input_ids, max_length=2)
      dec = [emo_model.tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label


def empathy_score(text):
  '''
  Computes a discrete numerical empathy score for an utterance (scale 0 to 2)
  '''
  with torch.no_grad():
      input_ids = emp_model.tokenizer.encode(text + '</s>', return_tensors='pt')
      output = emp_model.model.generate(input_ids=input_ids, max_length=2)
      dec = [emp_model.tokenizer.decode(ids) for ids in output]
  label = dec[0]
  if label == 'no':
    score = 0.0
  elif label == 'weak':
    score = 1.0
  else:
    score = 2.0
  #normalise between 0 and 1 dividing by the highest possible value:
  return score/2


def perplexity(sentence):
  '''
  Computes the PPL of an utterance using GPT2 LM
  '''
  tokenize_input = gpttokenizer.encode(sentence)
  tensor_input = torch.tensor([tokenize_input])
  with torch.no_grad():
      loss = gptmodel(tensor_input, labels=tensor_input)[0]
  return np.exp(loss.detach().numpy())


def repetition_penalty(sentence):
  '''
  Adds a penalty for each repeated (stemmed) token in
  an utterance. Returns the total penalty of the sentence
  '''
  word_list = regextokenizer.tokenize(sentence.lower())
  filtered_words = [word for word in word_list if word not in stopwords.words('english')]
  stem_list = [stemmer.stem(word) for word in filtered_words]
  penalty = 0
  visited = []
  for w in stem_list:
    if w not in visited:
      visited.append(w)
    else:
      penalty += 0.01
  return penalty


def fluency_score(sentence):
  '''
  Computes the fluency score of an utterance, given by the
  inverse of the perplexity minus a penalty for repeated tokens
  '''
  ppl = perplexity(sentence)
  penalty = repetition_penalty(sentence)
  score = (1 / ppl) - penalty
  #normalise by the highest possible fluency computed on the corpus:
  normalised_score = score / 0.16
  if normalised_score < 0:
    normalised_score = 0
  return round(normalised_score, 2)


def get_distance(s1, s2):
  '''
  Computes a distance score between utterances calculated as the overlap
  distance between unigrams, plus the overlap distance squared over bigrams,
  plus the overlap distance cubed over trigrams, etc (up to a number of ngrams
  equal to the length of the shortest utterance)
  '''
  s1 = re.sub(r'[^\w\s]', '', s1.lower()) #preprocess
  s2 = re.sub(r'[^\w\s]', '', s2.lower())
  s1_ws = regextokenizer.tokenize(s1) #tokenize to count tokens later
  s2_ws = regextokenizer.tokenize(s2)
  max_n = len(s1_ws) if len(s1_ws) < len(s2_ws) else len(s2_ws) #the max number of bigrams is the number of tokens in the shorter sentence
  ngram_scores = []
  for i in range(1, max_n+1):
    s1grams = nltk.ngrams(s1.split(), i)
    s2grams = nltk.ngrams(s2.split(), i)
    ngram_scores.append((td.overlap.normalized_distance(s1grams, s2grams))**i) #we normalize the distance score to be a value between 0 and 10, before raising to i
  normalised_dis = sum(ngram_scores)/(max_n) #normalised
  return normalised_dis


def compute_distances(sentence, dataframe):
  '''
  Computes a list of distances score between an utterance and all the utterances in a dataframe
  '''
  distances = []
  for index, row in dataframe.iterrows():
    df_s = dataframe['sentences'][index] #assuming the dataframe column is called 'sentences'
    distance = get_distance(df_s.lower(), sentence)
    distances.append(distance)
  return distances


def novelty_score(sentence, dataframe):
  '''
  Computes the mean of the distances beween an utterance
  and each of the utterances in a given dataframe
  '''
  if dataframe.empty:
    score = 1.0
  else:
    d_list = compute_distances(sentence, dataframe)
    d_score = sum(d_list)
    score = d_score / len(d_list)
  return round(score, 2)


def get_sentence_score(sentence, dataframe):
  '''
  Calculates how fit a sentence is based on its weighted empathy, fluency
  and novelty values
  '''
  empathy = empathy_score(sentence)
  fluency = fluency_score(sentence)
  novelty = novelty_score(sentence, dataframe)
  score = empathy + 0.75*fluency + 2*novelty
  return score
