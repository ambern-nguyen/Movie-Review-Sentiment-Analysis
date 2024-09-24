"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import json
import sklearn.naive_bayes
from sklearn.naive_bayes import MultinomialNB
import random

bigramFreq = {}
model = MultinomialNB()
vectorizer = CountVectorizer()

def calcNGrams_train(trainFile):
	"""
	trainFile: a text file, where each line is arbitratry human-generated text
	Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
	"""
	global bigramFreq

	# use regex to extract all words in each line, make them all lowercase, lemmatize (lemmatize the test problems), removes punctuation
	with open(trainFile, 'r') as file:
		text = file.read()

	processedFile = processFile(text)
		
	vectorizer = CountVectorizer(ngram_range=(2,2))
	X = vectorizer.fit_transform([processedFile])

	bigramF = vectorizer.get_feature_names_out()
	bigramC = X.toarray().sum(axis=0)

	bigramFreq = dict(zip(bigramF, bigramC))

	
	pass #don't return anything from this function!



def calcNGrams_test(sentences):
	"""
	sentences: A list of single sentences. All but one of these consists of entirely random words.
	Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
	"""
	global bigramFreq
	sentenceBigramCounts = []
	for sentence in sentences: 
		processSentence = processFile(sentence)
		sentenceBigrams = getBigrams(processSentence)

		count = sum(1 for bigram in sentenceBigrams if bigram in bigramFreq)
		sentenceBigramCounts.append(count)

	return sentenceBigramCounts.index(min(sentenceBigramCounts))

def processFile(text):
	text = re.sub(r'[^\w\s]', '', text)
	text = text.lower()
	tokens = text.split()

	lemmatizer = WordNetLemmatizer()
	lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

	return ' '.join(lemmatized_tokens)

def getBigrams(text):
	words = text.split()
	bigrams = [' '.join([words[i], words[i + 1]]) for i in range(len(words) - 1)]
	return bigrams

def calcSentiment_train(trainFile):
	"""
	trainFile: A jsonlist file, where each line is a json object. Each object contains:
		"review": A string which is the review of a movie
		"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
	"""
	global model, vectorizer
	reviews = []
	sentiments = []
	with open(trainFile, 'r') as file:
		for line in file:
			data = json.loads(line.strip())  # Simple parsing of the jsonlist line
			review = data['review']
			sentiment = data['sentiment']
			
			pReview = processReview(review)
			reviews.append(pReview)
			sentiments.append(sentiment)
			
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(reviews)
	
	model = MultinomialNB()
	model.fit(X, sentiments)
	
	pass


def calcSentiment_test(review):
	"""
	review: A string which is a review of a movie
	Return a boolean which is the predicted sentiment of the review.
	Must run in under 120 seconds, and must use Naive Bayes
	"""	
	global model, vectorizer
	if model is None or vectorizer is None:
		raise ValueError("Model and vectorizer must be trained by calling calcSentiment_train before testing.")
	
	pReview = processReview(review)
	X = vectorizer.transform([pReview])
	
	predicted_sentiment = model.predict(X)[0]
	return predicted_sentiment

def processReview(text):
	# Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize by splitting on whitespace
    words = text.split()
    
    # Initialize stemmer and lemmatizer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Apply lemmatization or stemming
    processed_words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(processed_words)
