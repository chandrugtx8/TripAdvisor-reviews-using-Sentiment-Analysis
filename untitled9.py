# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:06:28 2023

@author: chandru
"""
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax



loaded_model = pickle.load(open('C:/Users/chandru/Downloads/trained_model.sav', 'rb'))

fn=r"C:\Users\chandru\Downloads\Part1.csv"

df = pd.read_csv(fn, encoding='ISO-8859-1')
df_neg = df.loc[df['Rating'] <30]
df_neg = df_neg.reset_index(drop = True)
df_five = df.loc[df['Rating'] == 50]
df_five = df_five.reset_index(drop = True)
df_pos= df_five.loc[:len(df_neg)]
 
df_all = pd.concat([df_neg,df_pos], axis=0)
df_all= df_all.reset_index(drop = True)

df_all['Sentiment'] = np.where(df_all["Rating"] == 50,'Postive','Negative')
df_all = df_all.sample(frac=1)
df_all= df_all.reset_index(drop = True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(df_all.Review, df_all.Sentiment)

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
x_train_vec= v.fit_transform(x_train)
x_test_vec= v.transform(x_test)

from sklearn import svm
clf_svm = svm.SVC(kernel = 'linear')
clf_svm.fit(x_train_vec, y_train)

rev=["This is place very bad"]
rev_vec = v.transform(rev)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

example =df['Review'][50]
example



def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict











def sentiment_analysis(model, input_data):
    if model == "SVM":
        input_data = v.transform(input_data)
        return clf_svm.predict(input_data)
    elif model == "Vader":
        sia = SentimentIntensityAnalyzer()
        input_data = " ".join(input_data)
        sentiment = sia.polarity_scores(input_data)
        return sentiment
    elif model == "Roberta":
        analysis = polarity_scores_roberta(input_data[0])
        return analysis
    
    
    
    
def main():
    # giving a title
    st.title('Sentiment Analysis on Reviews Web App')

    model = st.radio("Select a model", ["SVM", "Vader","Roberta"])
    Review = st.text_input('Enter your text')

    analysis = ''

    if st.button('Submit'):
        analysis = sentiment_analysis(model, [Review])

    st.success(analysis)


if __name__ == '__main__':
    main()


