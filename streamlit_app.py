import streamlit as st

import pickle  # to load the saved pickle files
import pandas as pd
import numpy as np
import string
import nltk  # natural language tool kit used for text processing
from nltk.corpus import stopwords  # text processing
import string
from nltk.stem.porter import PorterStemmer  # text processing
import pandas as pd

ps = PorterStemmer()
from xgboost import XGBClassifier

nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    y = []
    # tokenization
    text = nltk.word_tokenize(text)
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    # removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming applied on text
    for i in text:
        y.append(ps.stem(i))
    return y


# loading  both the models from respective directory
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mnb_spam_detector.pkl', 'rb'))
# streamlit app title
st.title("SMS Spam classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    print(type(transform_sms))
    transform_sms = np.array(transform_sms)

    vector_input = tfidf.transform(transform_sms.astype('str')).toarray()
    print(type(vector_input))
    print(vector_input)
    vector_input = pd.DataFrame(vector_input, columns=tfidf.get_feature_names_out())
    prediction = model.predict(vector_input)[0]
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
