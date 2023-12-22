from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def tokenize(preprocessed_text):
    tokenizer = Tokenizer(split=' ', oov_token='<OOV>')
    with open('app/model/word_index.json') as json_word_index:
        word_index = json.load(json_word_index)
    tokenizer.word_index = word_index

    sequences = tokenizer.texts_to_sequences(preprocessed_text)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=150)

    return padded_sequences
  

