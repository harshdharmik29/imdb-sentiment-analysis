import streamlit as st
import pickle
import numpy as np

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# =========================
# Load Model and Vectorizer
# =========================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:20px;
}

.result-positive{
    padding:15px;
    border-radius:10px;
    background-color:#d4edda;
    color:#155724;
    font-size:20px;
    font-weight:bold;
}

.result-negative{
    padding:15px;
    border-radius:10px;
    background-color:#f8d7da;
    color:#721c24;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(
    '<div class="main-title">🎬 IMDB Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze Movie Reviews using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📌 Project Info")

    st.write("**Model:** Logistic Regression")
    st.write("**Vectorization:** TF-IDF")
    st.write("**Dataset:** IMDB Movie Reviews")
    st.write("**Task:** Sentiment Analysis")

    st.markdown("---")

    st.write("### Labels")
    st.success("Positive 😊")
    st.error("Negative 😞")

# =========================
# Input
# =========================
review = st.text_area(
    "Enter Movie Review",
    height=180,
    placeholder="Example: This movie was amazing and the acting was excellent..."
)

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        review_vector = tfidf.transform([review])

        prediction = model.predict(review_vector)[0]

        confidence = np.max(
            model.predict_proba(review_vector)
        ) * 100

        st.markdown("### Result")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-positive">
                😊 Positive Review
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-negative">
                😞 Negative Review
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(int(confidence))

        st.write(
            f"**Confidence Score:** {confidence:.2f}%"
        )

# =========================
# Footer
# =========================
st.markdown("---")

st.markdown(
    """
    <center>
    Built with ❤️ using Python, NLP, Scikit-Learn and Streamlit
    </center>
    """,
    unsafe_allow_html=True
)
