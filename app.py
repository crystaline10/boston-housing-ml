import streamlit as st
import joblib

# Step 1: Created a new file / script.
# Step 2: Setup Basic App Layout

st.title('Movie Review Sentiment Analyzer')

st.write(
    "First, choose a movie to review."
    " Then this app will predict whether the review"
     " is positive or negative overall."
)

# Step 3: Load the saved model

@st.cache_data
def load_model():
    model = joblib.load("sentiment_model.pkl")
    return model


# Load the model
model = load_model()

# Step 4: Create User Input Interface

user_movie_rev = st.text_area(
    "Enter a movie review to analyze:",
    height=75
)

button_analyze = st.button("Analyze")

# Step 5: Make Predictions & Display Results

if button_analyze:
    if user_movie_rev == '':
        st.warning("Don't forget to choose a movie review to analyze!")
    else:
        pred = model.predict([user_movie_rev])[0]
        prob = model.predict_proba([user_movie_rev])[0]

        classes = list(model.classes_).index(pred)
        confidence = prob[classes]

        st.subheader("Predictions")

        if pred == 'positive':
            st.success(f"Predicted Sentiment: Positive 👍")
        else:
            st.error(f"Predicted Sentiment: Negative 👎")
        
        st.subheader("Probability")

        st.write(f"Probability: {confidence:.2%}")

