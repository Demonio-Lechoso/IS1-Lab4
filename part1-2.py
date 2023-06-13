import numpy as np
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from os.path import exists

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

#cleaning text function, takes an preferably unclean text as parameter
#Tokenization is used in NLP to split paragraphs and sentences into smaller units that can be more easily assigned meaning.
def clean(uncleanedText):
#initializing the tokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    # initializing and importing english stopwords as well as importing stemmer class
    stopWordsEng = stopwords.words('english')
    stemmer = PorterStemmer()
    # clean text and clean stemmed text lists
    cleanedText = []
    cleanedAndStemmedText = [] 

    # remove stock market notations "$AAPL", hashtags, hyperlinks,strip the "\n" characters , retweet text "RT"
    #Basically, removing special characters
    uncleanedText = re.sub(r'\$\w*', "", uncleanedText)
    uncleanedText = re.sub(r'#', '', uncleanedText)
    uncleanedText = re.sub(r'https?:\/\/.*[\r\n]*', '', uncleanedText)
    uncleanedText = uncleanedText.strip()
    uncleanedText = re.sub(r'^RT[\s]+', '', uncleanedText)

    # tokenize the treated text
    uncleanedText = tokenizer.tokenize(uncleanedText)

    # Remove punctuation and stopwords by looping the words in the uncleanedText text
    # Basically remove the stop words in the text
    for word in uncleanedText:
        if (word not in stopWordsEng and  word not in string.punctuation):
            cleanedText.append(word)

    # Stemming the words, then appending them to the list of clean stemmed text, by looping through every word in the clean text list
    for word in cleanedText:
        stemmedWord = stemmer.stem(word)  # stemming word
        cleanedAndStemmedText.append(stemmedWord)  # append to the list

    # finally return clean stemmed text
    return cleanedAndStemmedText 

#function to clean a csv text file, save the cleaned text in a new csv file and return it 
def cleanedText():
    if exists('cleanedTweets.csv'): 
        df = pd.read_csv(r'cleanedTweets.csv')
        return df.sample(frac = 1) 
    else: 
        #read the news csv file
        df = pd.read_csv(r'Tweets.csv', names=['target','ids','date','flag','user','text'])
        df["text"]=df["text"].apply(clean)
        #save the cleaned text in the new cleaned news csv file
        df.to_csv('cleanedTweets.csv', index=False)
        return df.sample(frac = 1)
    
df = cleanedText()
df = df[:100000]

x = df['text'] 
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#convert text into numerical values in order to be used in the model
tf_vectorizer = TfidfVectorizer(use_idf=True)
x_train = tf_vectorizer.fit_transform(x_train).toarray()
x_test = tf_vectorizer.transform(x_test).toarray()

# Creating an SVM classifier
clf = svm.SVC(kernel='linear')

# Training the classifier on the training data
clf.fit(x_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(x_test)

# Calculating the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)