import os
import nltk
import re
from gensim import corpora, models, similarities
from cleaning import clean

def train():
	#Loads the data from the local storage
	synopses = []
	for filename in os.listdir('cnn-stories'):
		with open('cnn-stories/' + filename, 'r') as infile:
			synopses.append(infile.read())

	#Cleans the data
	corpus, dictionary = clean(synopses)

	#Saves the model and the dictionary in local storage
	corpora.Dictionary.save(dictionary, 'dictionary.dict')
	lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary, update_every=5, chunksize=10000, passes=100)
	lda.save('lda.model')

if __name__ == "__main__":
	train()