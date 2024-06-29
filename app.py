import streamlit as st
import json
import string
import nltk
import pandas as pd 
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from matplotlib import pyplot as plt
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title('Fake news Detector')
st.subheader('I can help you to classify fake news',divider='rainbow')
with st.container(height=250):
    news = st.text_area('Enter sample news',)
    but= st.button('Check')

class Preprocessing:
  def __init__(self,data):
    self.data = data
  def cleaning(self):
    stop = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    self.data= self.data.lower()
    self.data = word_tokenize(self.data)
    self.data = [i for i in self.data if i.isalpha()]
    self.data = [i for i in self.data if i not in string.punctuation]
    self.data = [i for i in self.data if i not in stop]
    self.data = [stemmer.stem(i) for i in self.data]
    self.data = " ".join(self.data)
    return self.data

def clean_apply(data):
   pre = Preprocessing(data)
   return pre.cleaning()

vect = pickle.load(open("vec.sav","rb"))
model = pickle.load(open("news_app.sav","rb"))
news_c = clean_apply(news)
x = vect.transform([news_c]).toarray()
y = model.predict(x)

if but:
    if y == 0:
        st.subheader('The news is fake ❌')
    else:
        st.subheader("It's real news you can trust it✅")

