from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
# Загрузка модели
model = load_model('models/RNN.h5')

def read_text(filename):
    with open(filename, mode='rt', encoding='utf-8') as file:
        text = file.read()
        sents = text.strip().split('\n')
        return [i.split('\t') for i in sents]

data = read_text("data/input_text.txt")
deu_eng = np.array(data)
deu_eng = deu_eng[:30000,:]
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1 
eng_length = 8 
deu_tokenizer = Tokenizer()
deu_tokenizer.fit_on_texts(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1 
deu_length = 8 

def get_word(n, tokenizer):
    if n == 0:
        return ""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return ""

def encode_sequences(tokenizer, length, lines):          
    # целочисленные кодирующие последовательности
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences со значениями 0
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

phrs_enc = encode_sequences(eng_tokenizer, eng_length, ["they are my cousins", "tom likes apples", "she wrote a new book", "read the story", "please open the door"])
print("phrs_enc:", phrs_enc.shape)

preds = np.argmax(model.predict(phrs_enc), axis=-1)
print("Preds:", preds.shape)
# print(preds[0])
print(get_word(preds[0][0], deu_tokenizer), get_word(preds[0][1], deu_tokenizer), get_word(preds[0][2], deu_tokenizer), get_word(preds[0][3], deu_tokenizer))
# print(preds[1])
print(get_word(preds[1][0], deu_tokenizer), get_word(preds[1][1], deu_tokenizer), get_word(preds[1][2], deu_tokenizer))
# print(preds[2])
print(get_word(preds[2][0], deu_tokenizer), get_word(preds[2][1], deu_tokenizer), get_word(preds[2][2], deu_tokenizer), get_word(preds[2][3], deu_tokenizer),get_word(preds[2][4], deu_tokenizer))
# print(preds[3])
print(get_word(preds[3][0], deu_tokenizer),get_word(preds[3][1], deu_tokenizer),get_word(preds[3][2], deu_tokenizer))
# print(preds[4])
print(get_word(preds[4][0], deu_tokenizer),get_word(preds[4][1], deu_tokenizer),get_word(preds[4][2], deu_tokenizer),get_word(preds[4][3], deu_tokenizer))