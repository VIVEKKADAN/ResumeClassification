import pickle
import streamlit as st
import pandas as pd
import re
import docx2txt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from docx import Document
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
#import wordcloud
#from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

svm_model_final = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

st.set_page_config(page_title="Resume classification", layout="wide")

st.title('Resume classification')

uploaded = st.file_uploader('Upload your Resume', type=["pdf", "docx", "doc", "txt"])

if uploaded is not None:
    if uploaded.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        document = Document(uploaded)
        paragraphs = [p.text for p in document.paragraphs]
        text = ' '.join(paragraphs)
    else:
        text = uploaded.getvalue().decode('latin-1')
    df = pd.DataFrame({'Resume': [text]})

    # regular expressions for phone number and email
    phone_regex = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # search for phone number and email in the resume text
    #phone_number = re.search(phone_regex, text)
    #email = re.search(email_regex, text)

    custom_stopwords = set(stopwords.words('english'))

    def clean_resume_text(resume_text):
        """
        This function takes in a string of text (resume) as input and returns a cleaned version of the text.
        """
        # Convert to lowercase
        resume_text = resume_text.lower()

        # Remove numbers and special characters
        resume_text = re.sub('[^a-zA-Z]', ' ', resume_text)

        # Remove punctuation
        resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespaces
        resume_text = ' '.join(resume_text.split())

        # Remove words with two or one letter
        resume_text = ' '.join(word for word in resume_text.split() if len(word) > 2)

        # Remove stop words
        resume_text = ' '.join(word for word in resume_text.split() if word not in custom_stopwords)

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        resume_text = ' '.join(lemmatizer.lemmatize(word) for word in resume_text.split())

        return resume_text

    df["clean_text"] = df["Resume"].apply(clean_resume_text)
    all_resume_text = ' '.join(df["clean_text"])

    all_words = all_resume_text.split()
    word_counts = Counter(all_words)

    # Clean the text by removing short words and noise words
    noise_words = ['a','sport','family','married']

    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'\b\w{{1,2}}\b|\b(?:{})\b'.format('|'.join(noise_words)), '', x))

    resume = df.loc[:, 'clean_text']

    X_train_tfidf = tfidf_vectorizer.transform(resume).toarray()

    y = svm_model_final.predict(X_train_tfidf)


    if y == 0:
        st.subheader("Candidate's Resume Matches PeopleSoft Category")
    elif y == 1:
        st.subheader("Candidate's Resume Matches ReactJS Category")
    elif y == 2:
        st.subheader("Candidate's Resume Matches SQL Developer Category")
    else:
        st.subheader("Candidate's Resume Matches Workday Category")

    #if phone_number is not None:
     #   st.write("Phone number:", phone_number.group(0))
    #else:
     #   st.write("No phone number found")

    #if email is not None:
     #   st.write("Email:", email.group(0))
    #else:
     #   st.write("No email found")
    #if st.button('WordCloud:'):
        # Define a function to plot word cloud
     #   def plot_cloud(wordcloud):
            # Set figure size
                # Set figure size
      #          plt.figure(figsize=(40, 30))
      #          # Display image
      #          plt.imshow(wordcloud)
                # No axis details
      #          plt.axis("off");

       #         words = np.array(df['clean_text'])
       #         words = words.astype(str)  # Convert to string type
#
                # Generate word cloud
       #         wordcloud = WordCloud(width=3000, height=2000, background_color='black', max_words=100, colormap='Set3',
       #                               stopwords=stopwords).generate(' '.join(words))

                # Plot the word cloud
         #       plt.figure(figsize=(10, 6))
         #       plt.imshow(wordcloud, interpolation='bilinear')
         #       plt.axis('off')
           #     plt.show()

    if st.button('Developed by:'):


            st.subheader('P-257 Group -1,Team Members')
            st.write('Abhishek Singh')
            st.write('Anamika Birari')
            st.write('Saipallavi Prakhya')
            st.write('Vaibhav Dongre')
            st.write('Venkat Pulavarthi')
            st.write('Vivek K')
            st.write('Vivek Chaudhary')

            st.subheader("We have used SVM Classifier Model to predict the Categories of the Resumes")
            st.image("https://th.bing.com/th/id/OIP.J7JOUuPueCo12jp1BQl_hAHaE8?pid=ImgDet&rs=1")


    st.subheader("🧑‍💻 Thank you! 👋")
