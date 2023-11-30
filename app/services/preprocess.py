import nltk
import string
import re
import ast
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

def slang_word(text):
  with open("app/model/slang_words.txt", "r") as slang_file:
    slang_content = slang_file.read()
    slang_words = ast.literal_eval(slang_content)

  filtered = []
  for txt in text:
    if txt not in slang_words.keys():
      filtered.append(txt)
    if txt in slang_words.keys():
      x = txt.replace(txt, slang_words[txt])
      filtered.append(x)
  text = filtered
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
  cleaned_text = [remove_punctuation(x) for x in text]
  cleaned_text = [case_folding(x) for x in cleaned_text]
  tokenized_text = [tokenizingText(x) for x in cleaned_text]
  slang_words = [slang_word(x) for x in tokenized_text]
  stemmed_text = [stemmingText(x) for x in slang_words]
  preprocessed_text = [toSentence(x) for x in stemmed_text]

  return preprocessed_text