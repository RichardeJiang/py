import keras
import numpy as np 
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Conv2D, Embedding, Reshape
from keras.layers.merge import Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from gensim.models import Word2Vec
from collections import Counter
from wv import loadData, processStr

def obtainData(typeOfData = "train"):
	fileName = "20_train" if typeOfData == "train" else "20_test"
	sentences, emojis = loadData(fileName)

	texts = [' '.join(ele) for ele in sentences]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	wordIdMap = tokenizer.word_index
	data = pad_sequences(sequences, padding='post', truncating='post')

	labels = to_categorical(np.asarray(emojis))

	return data, labels, wordIdMap

def buildDataFull():
	fileNames = ["20_train", "20_validation", "20_test"]
	sentences = []
	emojis = []
	trainLength  = 0
	testLength = 0
	for ele in fileNames:
		currS, currE = loadData(ele)
		if trainLength is 0:
			trainLength = len(currE)
		elif testLength is 0:
			testLength = len(currE)
		sentences += currS
		emojis += currE

	emojiCounts = Counter(emojis)
	print "number of emojis detected is: " + str(len(emojiCounts))
	emojiIdMap = {}
	for index, emoji in enumerate(emojiCounts):
		emojiIdMap[emoji] = index
	emojiLabels = [emojiIdMap[ele] for ele in emojis]
	texts = [' '.join(ele) for ele in sentences]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	wordIdMap = tokenizer.word_index
	idWordMap = {i : w for w, i in wordIdMap.items()}
	data = pad_sequences(sequences, padding='post', truncating='post')
	maxLength = data.shape[1]

	labels = to_categorical(np.asarray(emojiLabels))

	trainX = data[:trainLength]
	testX = data[-testLength:]

	trainY = labels[:trainLength]
	testY = labels[-testLength:]

	return trainX, testX, trainY, testY, wordIdMap, maxLength

if __name__ == "__main__":
	filterSizes = [3, 4, 5]
	numOfFilters = 10
	dropout = 0.5
	batchSize = 50
	epochs = 20
	sequenceLength = 20 # Twitter max length is 140 chars
	embeddingDim = 50
	numOfLabels = 20
	drop = 0
	wvModel = Word2Vec.load('vectors.bin')
	# sentencesTrain, emojisTrain = obtainData()
	# dataTrain, labelsTrain, wordIdTrain = obtainData()
	# dataTest, labelsTest, wordIdTest = obtainData("test")
	dataTrain, dataTest, labelsTrain, labelsTest, wordIdMap, maxLength = buildDataFull()


	embeddingMatrix = np.zeros((len(wordIdMap) + 1, embeddingDim))
	for word, i in wordIdMap.items():
		try:
			vector = wvModel.wv[word]
			embeddingMatrix[i] = vector
		except:
			pass
		# vector = wvModel.wv[word]
		# if vector is not None:
		# 	embeddingMatrix[i] = vector

	# embeddingLayer = Embedding(len(wordIdMap),
	# 							50,  #word embedding dimension
	# 							weights=[embeddingMatrix],
	# 							input_length=maxLength,
	# 							trainable=False)

	sequenceInput = Input(shape=(maxLength, ), dtype='int32')
	embedding = Embedding(input_dim = len(wordIdMap), output_dim = embeddingDim, input_length = maxLength)(sequenceInput)
	# embeddedSequences = embeddingLayer(sequenceInput)
	finalEmbeddedSeq = Reshape((maxLength, embeddingDim, 1))(embedding)



	conv_0 = Conv2D(numOfFilters, kernel_size=(filterSizes[0], embeddingDim), padding='valid', kernel_initializer='normal', activation='relu')(finalEmbeddedSeq)
	conv_1 = Conv2D(numOfFilters, kernel_size=(filterSizes[1], embeddingDim), padding='valid', kernel_initializer='normal', activation='relu')(finalEmbeddedSeq)
	conv_2 = Conv2D(numOfFilters, kernel_size=(filterSizes[2], embeddingDim), padding='valid', kernel_initializer='normal', activation='relu')(finalEmbeddedSeq)

	maxpool_0 = MaxPooling2D(pool_size=(maxLength - filterSizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	maxpool_1 = MaxPooling2D(pool_size=(maxLength - filterSizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	maxpool_2 = MaxPooling2D(pool_size=(maxLength - filterSizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

	concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
	flatten = Flatten()(concatenated_tensor)
	dropout = Dropout(drop)(flatten)
	output = Dense(units = numOfLabels, activation = 'softmax')(dropout)

	model = Model(inputs=sequenceInput, outputs=output)

	# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	# adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
	print("Traning Model...")
	model.fit(dataTrain, labelsTrain, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=(dataTest, labelsTest))


	pass