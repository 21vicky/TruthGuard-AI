
import streamlit as st
import joblib

from text_cleaner import clean_text

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d1b, #101a30);
    color: #f1f5f9;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero h1 {
    font-size: 46px;
    color: #60a5fa;
    font-weight: 800;
}

.hero p {
    color: #a5b4fc;
    font-size: 17px;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.real {
    background: rgba(34,197,94,0.12);
    border: 1px solid #22c55e;
}

.fake {
    background: rgba(239,68,68,0.12);
    border: 1px solid #ef4444;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    text-align: center;
}

div.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 30px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("models/fake_news_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf


model, tfidf = load_model()


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🛡️ TruthGuard AI</h1>
    <p>AI-Powered Fake News Detection System</p>
    <p>Analyze news content using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ---------------- EXAMPLE NEWS ----------------

st.subheader("📰 Try an Example")

col1, col2 = st.columns(2)

real_example = """
The White House announced a new policy initiative on Friday.
Officials said the proposal aims to support businesses and
create new opportunities for entrepreneurs across the country.
"""

fake_example = """
SHOCKING! A secret miracle drink has been discovered that
cures every disease overnight. Scientists are hiding the truth
and governments do not want people to know about it!
"""

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

with col1:
    if st.button("🟢 Load Real News Example", use_container_width=True):
        st.session_state.news_input = real_example

with col2:
    if st.button("🔴 Load Fake News Example", use_container_width=True):
        st.session_state.news_input = fake_example


# ---------------- NEWS INPUT ----------------

st.subheader("✍️ Enter News Content")

news = st.text_area(
    "Paste a headline or complete news article:",
    key="news_input",
    height=220,
    placeholder="Paste your news article here..."
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze News", type="primary", use_container_width=True):

    if not news.strip():
        st.warning("Please enter some news content.")

    else:
        cleaned_news = clean_text(news)

        analyzer = tfidf.build_analyzer()
        tokens = analyzer(cleaned_news)

        vocabulary = tfidf.vocabulary_

        recognized_tokens = [
            token for token in tokens
            if token in vocabulary
        ]

        total_words = len(tokens)
        recognized_words = len(recognized_tokens)

        coverage = (
            recognized_words / total_words * 100
            if total_words > 0 else 0
        )

        news_vector = tfidf.transform([cleaned_news])

        st.subheader("📊 Input Quality")

        q1, q2, q3 = st.columns(3)

        q1.metric("Words Detected", total_words)
        q2.metric("Recognized Words", recognized_words)
        q3.metric("Vocabulary Coverage", f"{coverage:.1f}%")

        if total_words < 8:
            st.warning(
                "Please enter a longer article with at least 8 meaningful words."
            )

        elif coverage < 30 or news_vector.nnz == 0:
            st.warning(
                "This content contains too few words recognized by the model. "
                "Try entering a more detailed article."
            )

        else:
            prediction = model.predict(news_vector)[0]
            probabilities = model.predict_proba(news_vector)[0]

            scores = dict(zip(model.classes_, probabilities))

            real_score = scores[0] * 100
            fake_score = scores[1] * 100

            st.divider()
            st.subheader("🔎 Analysis Result")

            if prediction == 0:

                st.markdown("""
                <div class="result-card real">
                    <h1>🟢 REAL NEWS</h1>
                    <p>The model classified this content as Real News.</p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-card fake">
                    <h1>🔴 FAKE NEWS</h1>
                    <p>The model classified this content as Fake News.</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("### Model Score Distribution")

            score1, score2 = st.columns(2)

            with score1:
                st.metric("Real News Score", f"{real_score:.2f}%")
                st.progress(int(real_score))

            with score2:
                st.metric("Fake News Score", f"{fake_score:.2f}%")
                st.progress(int(fake_score))

            st.caption(
                "These scores represent the model's output distribution, "
                "not verified probabilities that the article is true or false."
            )

            with st.expander("ℹ️ How does TruthGuard AI work?"):
                st.write("""
                1. Cleans the submitted news content.
                2. Converts text into numerical TF-IDF features.
                3. Uses a Logistic Regression classifier.
                4. Displays the predicted category and model scores.
                """)

# ---------------- DISCLAIMER ----------------

st.divider()

st.warning(
    "⚠️ Disclaimer: TruthGuard AI is an educational machine learning "
    "project. It does not independently verify facts or replace "
    "professional fact-checking."
)

st.markdown("""
<div class="footer">
    Built with Python • Scikit-learn • TF-IDF • Streamlit
    <br>
    TruthGuard AI | Day 05 – 30 Days 30 AI Projects
</div>
""", unsafe_allow_html=True)
