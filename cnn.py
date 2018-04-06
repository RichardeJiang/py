import keras
import numpy as np 
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from gensim.models import Word2Vec
from wv import loadData, processStr

def obtainData(typeOfData = "train"):
	fileName = "20_train" if typeOfData == "train" else "20_test"
	sentences, emojis = loadData(fileName)

	texts = [' '.join(ele) for ele in sentences]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	data = pad_sequences(sequences, padding='post', truncating='post')

	labels = to_categorical(np.asarray(emojis))

	return data, labels

if __name__ == "__main__":
	filterSizes = [3, 4, 5]
	numOfFilters = 10
	dropout = 0.5
	batchSize = 50
	epochs = 20
	sequenceLength = 20 # Twitter max length is 140 chars
	wvModel = Word2Vec.load('vectors.bin')
	sentencesTrain, emojisTrain = obtainData()
	dataTrain, labelsTrain = obtainData()
	dataTest, labelsTest = obtainData("test")

	# len(wvModel.wv.vocab)
	
	# texts = [' '.join(ele) for ele in sentencesTrain]
	# tokenizer = Tokenizer()
	# tokenizer.fit_on_texts(texts)
	# sequences = tokenizer.texts_to_sequences(texts)
	# data = pad_sequences(sequences, padding='post', truncating='post')

	# labels = to_categorical(np.asarray(emojisTrain))
	# print('Shape of data tensor:', data.shape)
	# print('Shape of label tensor:', labels.shape)
	

	pass