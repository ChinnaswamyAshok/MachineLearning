# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:11:04 2020
This file includes k-means cluster model creation and prediction ENDTOEND
@author: Ashok Chinnaswamy 

"""

import pandas as pd
# Read raw data set from .xlsx 
RawData = pd.read_excel(r"C:\AshokC\splunk\input\LEA_Input_data.xlsx",dtype=str)

# Loading value column from the RawData
xtrain = RawData.iloc[:,20]	

from nltk.stem import PorterStemmer

# This method will convert numeric contents in sentence to hexadecimal string
def inttostring(word):
    if word.isdigit():
        value = int(word)
        c=hex(value)
        return c
    else:
        return word
    
# This method performs
#   * removal of punctuation
#   * converts lower case
#   * stemming
#   * int to string
#   * removal of stopwords
# return final list of preprocessed sentence
def preprocess(xtrain):
    final = []
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                 "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                 "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                 "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                 "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                 "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                 "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                 "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                 "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                 "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
                 "now"]

    for i in range(len(xtrain)):
        if isinstance(xtrain[i], str):
            words = xtrain[i].split()

            # remove punctuation from each word
            import string
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in words]
            # lower case
            wordslower = [word.lower() for word in stripped]

            # removal of stop words
            porter = PorterStemmer()
            tokens_without_sw = [porter.stem(word) for word in wordslower if not word in stopwords ]

            # stemming of words
            porter = PorterStemmer()
            stemmed = [inttostring(porter.stem(word)) for word in tokens_without_sw]
            final.append(' '.join(stemmed))
        else:
            words = xtrain[i]
            final.append(xtrain[i])
    return final


# Storing the preprocessed list
finalSentence = preprocess(xtrain)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# using TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=False)
inputs = vectorizer.fit_transform(finalSentence)
print (inputs)

# k-means clustering setting 10 cluster groups based on keywords
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(inputs)



order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# print the centroids into which clusters
for i in range(true_k):
    print(i)
    print('------------')
    for ind in order_centroids[i, :10]:
           print(terms[ind])
           
               
# Passing input into the model to find the predicted output
Input=['cursor is not working for picture']
InputSentence = preprocess(Input)
P = vectorizer.transform(InputSentence)
predicted = model.predict(P)
print(predicted)





