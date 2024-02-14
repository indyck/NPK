import string 
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers 
from sklearn.model_selection import train_test_split


# Читаем текстовый файл
def read_text(filename):
    with open(filename, mode='rt', encoding='utf-8') as file:
        text = file.read()
        sents = text.strip().split('\n')
        return [i.split('\t') for i in sents]

data = read_text("data/input_text.txt")
deu_eng = np.array(data)

deu_eng = deu_eng[:30000,:]
print("Dictionary size:", deu_eng.shape)

# Убираем пунктуацию
deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]] 
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]] 

# Переводим в нижный регистр
for i in range(len(deu_eng)): 
    deu_eng[i,0] = deu_eng[i,0].lower() 
    deu_eng[i,1] = deu_eng[i,1].lower()
    
# Подготавливаем английский токенайзер
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1 
eng_length = 8 

# Подготавливаем немецкий токенайзер
deu_tokenizer = Tokenizer()
deu_tokenizer.fit_on_texts(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1 
deu_length = 8 

# Кодируем и дополненяем последовательности
def encode_sequences(tokenizer, length, lines):          
    # целочисленные кодирующие последовательности
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences со значениями 0
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq
     
# Разделение на обучающие и тестируемые данные
train, test = train_test_split(deu_eng, test_size=0.2, random_state=12)

# Подготовка данных для обучения 
trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_sequences(deu_tokenizer, deu_length, train[:, 1])

# Подготовка тестовых данных 
testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_sequences(deu_tokenizer, deu_length, test[:, 1])

# Собрание модели
def make_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy')
    return model

print("deu_vocab_size:", deu_vocab_size, deu_length)
print("eng_vocab_size:", eng_vocab_size, eng_length)

# Компиляция модели
model = make_model(eng_vocab_size, deu_vocab_size, eng_length, deu_length, 512)

# Тренировка модели
num_epochs = 250
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=num_epochs, batch_size=512, validation_split=0.2, callbacks=None, verbose=1)
model.save('RNN.h5')


