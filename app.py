
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("IMDB Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        st.success("Positive Review")
    else:
        st.error("Negative Review")
