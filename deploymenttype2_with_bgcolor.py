import pandas as pd
import streamlit as st
import docx2txt
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import plotly.express as px
import random

stop = set(stopwords.words('english'))
import pickle

vectors = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

nltk.download('wordnet')
nltk.download('stopwords')

resume = []


def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    rem_email = re.sub(r'\S+@\S+', '', rem_num)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_email)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


def mostcommon_words(cleaned, i):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(cleaned)
    mostcommon = FreqDist(cleaned.split()).most_common(i)
    return mostcommon


def display_wordcloud(mostcommon):
    wordcloud = WordCloud(width=2000, height=1500, background_color='black').generate(str(mostcommon))
    a = px.imshow(wordcloud)
    st.plotly_chart(a)


def display_words(mostcommon_small):
    x, y = zip(*mostcommon_small)
    chart = pd.DataFrame({'keys': x, 'values': y})
    fig = px.bar(chart, x=chart['keys'], y=chart['values'], height=700, width=700)
    st.plotly_chart(fig)


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, pink, cyan);
        }
        .classification-section {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 10px;
            background-color: #FFC0CB;
            margin-bottom: 20px;
            text-align: center;
        }
        .classification-section h1 {
            color: black;
            margin: 0;
            padding: 10px 0;
            font-size: 56px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='classification-section'><h1>RESUME CLASSIFICATION</h1></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<span style='font-size: 24px; font-weight: bold;'>Upload Your Resume</span>",
        unsafe_allow_html=True
    )
    upload_file = st.file_uploader("", type=['docx', 'pdf', 'doc', 'txt'], accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename': [doc_file.name],
                                'filetype': doc_file.type.split('.')[-1].upper(),
                                'filesize': str(doc_file.size) + ' KB'}
                file_type = pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed = display(doc_file)

                cleaned = preprocess(display(doc_file))
                transformed_input = vectors.transform([cleaned]).toarray()  # Convert sparse matrix to dense array
                predicted = model.predict(transformed_input)

                if predicted == 0:
                    st.subheader("Candidate's Resume matches PeopleSoft category.")
                elif predicted == 1:
                    st.subheader("Candidate's Resume matches ReactJS category.")
                elif predicted == 2:
                    st.subheader("Candidate's Resume matches SQL Developer category.")
                else:
                    st.subheader("Candidate's Resume matches Workday category.")

                display_wordcloud(mostcommon_words(cleaned, 25))
                st.subheader("🧑‍💻 Thank you! 👋")
    if st.button('Developed by:'):
                    st.subheader('P-257 Group -1,Team Members')
                    st.write(' Abhishek Singh,', ' Anamika Birari,', ' Saipallavi Prakhya,', ' Vaibhav Dongre,', ' Venkat Pulavarthi,', ' Vivek Chaudhary,', ' Vivek K')
                    st.subheader("We have used SVM Classifier Model to predict the Categories of the Resumes")




if __name__ == '__main__':
    main()