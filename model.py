import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from sklearn.externals import joblib

NB_spam_model = open('NB_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)

clf.predict("Here is a offer for you")