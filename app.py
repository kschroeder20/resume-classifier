import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# Define the text normalization function (must match the training version)
def normalize_document(doc):
    """Normalize document text by removing special characters and stopwords"""
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    wpt = nltk.WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    stop_words = stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_resume_classifier_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the Jupyter notebook to train and save the model first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load resources
model = load_model()

# App title and description
st.title("📄 Resume Classifier")
st.markdown("""
This application classifies resumes into job categories based on their content.
Simply paste the resume text below and click **Classify** to see the predicted job category.
""")

# Text input area
resume_text = st.text_area(
    "Paste Resume Text Here:",
    height=300,
    placeholder="Copy and paste the resume text here..."
)

# Classify button
if st.button("Classify Resume", type="primary"):
    if resume_text.strip():
        with st.spinner("Analyzing resume..."):
            # Normalize the input text
            normalized_text = normalize_document(resume_text)

            # Make prediction
            prediction = model.predict([normalized_text])[0]

            # Get prediction probabilities if available
            try:
                probabilities = model.predict_proba([normalized_text])[0]
                classes = model.classes_

                # Display results
                st.success("Classification Complete!")

                # Main prediction
                st.markdown(f"### Predicted Category: **{prediction}**")

                # Show confidence scores
                st.markdown("#### Confidence Scores:")

                # Sort probabilities in descending order
                sorted_indices = np.argsort(probabilities)[::-1]

                for idx in sorted_indices:
                    confidence = probabilities[idx] * 100
                    category = classes[idx]

                    # Create a progress bar for each category
                    if category == prediction:
                        st.markdown(f"**{category}**")
                    else:
                        st.markdown(f"{category}")
                    st.progress(probabilities[idx])
                    st.caption(f"{confidence:.2f}%")

            except AttributeError:
                # If model doesn't support predict_proba (e.g., LinearSVC)
                st.success("Classification Complete!")
                st.markdown(f"### Predicted Category: **{prediction}**")
                st.info("Confidence scores are not available for this model type.")
    else:
        st.warning("Please paste some resume text before classifying.")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This resume classifier uses machine learning to automatically categorize resumes into job domains.

    **How it works:**
    1. Paste resume text into the text box
    2. Click the "Classify Resume" button
    3. View the predicted job category and confidence scores

    **Categories include:**
    - Data Science
    - HR
    - Designer
    - Software Engineering
    - Marketing
    - And more...
    """)

    st.header("Tips")
    st.markdown("""
    - Include complete resume text for best results
    - The model analyzes skills, experience, and keywords
    - Longer, more detailed resumes typically yield better predictions
    """)

    st.markdown("---")
    st.caption("Resume Classifier v1.0")
