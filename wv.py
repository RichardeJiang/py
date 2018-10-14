from gensim.models import Word2Vec
import string

def loadData(fileName = "20_train"):
	tf = open(fileName, "r")
	sentences = []
	emojis = []
	for line in tf:
		x = line.split("\t")[0]
		y = line.split("\t")[1]
		x = processStr(x)
		sentences.append(x.split())
		emojis.append(y)
	return sentences, emojis

def processStr(inputStr):
	result = inputStr.replace("@user", "").translate(None, string.punctuation).translate(None, string.digits).lower()
	return result

if __name__ == "__main__":
	sentences,  = loadData()
	model = Word2Vec(sentences, min_count = 5, size = 50, workers = 6, sg = 0, iter = 50)
	model.save('vectors.bin')
	pass