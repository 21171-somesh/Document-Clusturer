import string
import nltk
import re
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models, similarities 

#Removes proper nouns from the text file
def strip_proppers(text):
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
	return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

#Tokenizes and Stems the input text
def tokenize_and_stem(text):
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	stems = [stemmer.stem(t) for t in filtered_tokens]
	return stems

#Load necessary files
stemmer = SnowballStemmer("english")
stopwords = dict()
with open('stopwords.txt', 'rU') as infile:
	for line in infile:
		stopwords[line[:-1]] = 1

#Main cleaning function
def clean(synopses):
	preprocess = [strip_proppers(doc) for doc in synopses]
	tokenized_text = [tokenize_and_stem(text) for text in preprocess]
	texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

	dictionary = corpora.Dictionary(texts)
	dictionary.filter_extremes(no_below = 1, no_above = 0.8)
	corpus = [dictionary.doc2bow(text) for text in texts]

	return corpus, dictionary

