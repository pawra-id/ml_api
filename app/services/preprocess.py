import nltk
import string
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')

def remove_punctuation(text):
  text = re.sub('-',' ',text)
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

def case_folding(text):
  text = text.lower()
  return text

def tokenizingText(text):
  text = word_tokenize(text)
  return text

def stemmingText(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  text = [stemmer.stem(word) for word in text]
  return text

def toSentence(list_words): # Convert list of words into sentence
  sentence = ' '.join(word for word in list_words)
  return sentence


def data_preprocessing(text):
  cleaned_text = remove_punctuation(text)
  cleaned_text = case_folding(cleaned_text)
  tokenized_text = tokenizingText(cleaned_text)
  stemmed_text = stemmingText(tokenized_text)
  preprocessed_text = toSentence(stemmed_text)

  return preprocessed_text