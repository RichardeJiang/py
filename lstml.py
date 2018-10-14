import keras
import numpy as np 
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Conv2D, Embedding, Reshape, LSTM, Bidirectional
from keras.layers.merge import Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from gensim.models import Word2Vec
from collections import Counter
from wv import loadData, processStr
import json

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
	fileNames = ["20_train", "20_test"]
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
	idEmojiMap = {}
	for index, emoji in enumerate(emojiCounts):
		emojiIdMap[emoji] = index
		idEmojiMap[index] = emoji
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

	return trainX, testX, trainY, testY, wordIdMap, maxLength, idEmojiMap

def getEmojiNVocab():
	packedData = json.load(open('data.json'))
	idEmojiMap = packedData["emo"]
	wordIdMap = packedData["dic"]
	maxLength = packedData["len"]

	emojiTokenMap = {}
	emojiTokenMap["eoji2764"] = ":heart:"
	emojiTokenMap["eoji2744"] = ":snowflake:"
	emojiTokenMap["eoji1f4aa"] = ":muscle:"
	emojiTokenMap["eoji1f60d"] = ":heart_eyes:"
	emojiTokenMap["eoji2728"] = ":sparkles:"
	emojiTokenMap["eoji1f62d"] = ":sob:"
	emojiTokenMap["eoji1f618"] = ":kissing_heart:"
	emojiTokenMap["eoji1f60e"] = ":sunglasses:"
	emojiTokenMap["eoji1f44c"] = ":ok_hand:"
	emojiTokenMap["eoji1f602"] = ":joy:"
	emojiTokenMap["eoji1f525"] = ":fire:"
	emojiTokenMap["eoji1f64f"] = ":pray:"
	emojiTokenMap["eoji1f499"] = ":blue_heart:"
	emojiTokenMap["eoji1f4af"] = ":100:"
	emojiTokenMap["eoji1f495"] = ":two_hearts:"
	emojiTokenMap["eoji1f60a"] = ":blush:"
	emojiTokenMap["eoji1f64c"] = ":raised_hands:"
	emojiTokenMap["eoji1f389"] = ":tada:"
	emojiTokenMap["eoji1f48b"] = ":kiss:"
	emojiTokenMap["eoji1f384"] = ":christmas_tree:"

	return idEmojiMap, wordIdMap, maxLength, emojiTokenMap

def getXData(maxLength, wordIdMap, inputText):

	# packedData = json.load(open('data.json'))
	# maxLength = int(packedData["len"])
	# wordIdMap = packedData["dic"]
	# idEmojiMap = packedData["emo"]

	seq = [wordIdMap[ele] for ele in inputText if ele in wordIdMap]
	seq = [seq]
	# seq = tk.texts_to_sequences(inputText)
	data = pad_sequences(seq, maxlen = maxLength, padding='post', truncating='post')
	return data

def makePrediction(inputStr, maxLength, wordIdMap, model):
	processedStr = processStr(inputStr)
	data = getXData(maxLength, wordIdMap, processedStr)
	predicted = model.predict(data)[0]
	predictedY = predicted.argmax(axis=-1)

	return predictedY

if __name__ == "__main__":
	filterSizes = [3, 4, 5]
	numOfFilters = 50    # tested with 10, 20
	dropout = 0.5
	batchSize = 500
	epochs = 20
	sequenceLength = 20 # Twitter max length is 140 chars
	embeddingDim = 50
	numOfLabels = 20
	drop = 0.5
	wvModel = Word2Vec.load('vectors300.bin')
	# sentencesTrain, emojisTrain = obtainData()
	# dataTrain, labelsTrain, wordIdTrain = obtainData()
	# dataTest, labelsTest, wordIdTest = obtainData("test")
	dataTrain, dataTest, labelsTrain, labelsTest, wordIdMap, maxLength, idEmojiMap = buildDataFull()
	packedData = {"len": maxLength, "dic": wordIdMap, "emo": idEmojiMap}
	js = json.dumps(packedData)
	fp = open("datalstml.json", "w")
	fp.write(js)
	fp.close()

	embeddingMatrix = np.zeros((len(wordIdMap)+1, embeddingDim))
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
	embedding = Embedding(input_dim = len(wordIdMap) + 1, 
						output_dim = embeddingDim, 
						weights = [embeddingMatrix],
						input_length = maxLength,
						trainable = False)(sequenceInput)
	# embeddedSequences = embeddingLayer(sequenceInput)
	finalEmbeddedSeq = Reshape((maxLength, embeddingDim, 1))(embedding)


	model = Sequential()
	model.add(Embedding(len(wordIdMap) + 1, embeddingDim, weights = [embeddingMatrix], input_length=maxLength))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(64, return_sequences = True)))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(64)))
	model.add(Dropout(0.2))
	model.add(Dense(units = numOfLabels, activation='softmax'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


	# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
	print("Traning Model...")
	model.fit(dataTrain, labelsTrain, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=(dataTest, labelsTest))
	model.save('emolstml.h5')

	pass