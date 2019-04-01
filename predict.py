from gensim import models, corpora
import numpy
import re
from cleaning import strip_proppers, tokenize_and_stem, stopwords, stemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def formatList(li):
	s = li[1]
	s = str(s)
	li = []
	t = str()
	flag = 0
	for i in s:
		if flag == 2:
			li.pop()
			flag = 0
			continue
		if i.isalpha():
			if i == 'n':
				flag = 1
			t += i
		else:
			if(flag == 1 and i == '\''):
				flag += 1
			if t != "":
				li.append(t)
			t = ""
	return li

def pred(text):
	#Loading the files in the program
	question = text;
	model = models.LdaModel.load('lda.model')
	dictionary = corpora.Dictionary.load('dictionary.dict')
	stopwords = {}
	with open('stopwords.txt', 'rU') as infile:
		for line in infile:
			stopwords[line[:-1]] = 1

	#Cleaning up of the text which we want to predict
	words = []
	sentences = nltk.sent_tokenize(question.lower())
	for sentence in sentences:
		tokens = nltk.word_tokenize(sentence)
		text = [word for word in tokens if word not in stopwords]
		tagged_text = nltk.pos_tag(text)
		for word, tag in tagged_text:
			words.append({"word": word, "pos": tag})
	lem = WordNetLemmatizer()
	nouns = []
	for word in words:
		if word["pos"] in ["NN", "NNS"]:
			nouns.append(lem.lemmatize(word["word"]))

	#Predicting		
	corpus = dictionary.doc2bow(nouns)
	topic_vec = model[corpus]

	#Finding most related topic
	maxEle, pos = 0, 0
	for i, j in topic_vec:
		if(j > maxEle):
			maxEle = j
			pos = i

	#Printing Stuff
	print(topic_vec)
	li = list()
	for i in model.show_topics()[pos]:
		li.append(i)

	return formatList(li)