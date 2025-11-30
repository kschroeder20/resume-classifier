import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

#####################################
########## Helper Functions #########
#####################################

# Text normalization
# Remove special characters and stopwords
# Tokenize using nltk
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    
    # tokenize document
    wpt = nltk.WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    
    # remove stopwords
    stop_words = stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stop_words]

    doc = ' '.join(filtered_tokens)
    return doc


#####################################
########### Load Model ##############
#####################################

@st.cache_resource
def load_model():
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_resume_classifier_model.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.error("Please run the Jupyter notebook to train and save the model first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

#####################################
########### Streamlit App ###########
#####################################

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
            normalized_text = normalize_document(resume_text)
            prediction = model.predict([normalized_text])[0]

            # get prediction probabilities
            try:
                probabilities = model.predict_proba([normalized_text])[0]
                classes = model.classes_

                # display results
                st.success("Classification Complete!")
                st.markdown(f"### Predicted Category: **{prediction}**")
                st.markdown("#### Confidence Scores:")

                sorted_indices = np.argsort(probabilities)[::-1]

                for idx in sorted_indices:
                    confidence = probabilities[idx] * 100
                    category = classes[idx]

                    # progress bar for each category
                    if category == prediction:
                        st.markdown(f"**{category}**")
                    else:
                        st.markdown(f"{category}")
                    st.progress(probabilities[idx])
                    st.caption(f"{confidence:.2f}%")

            except AttributeError:
                st.success("Classification Complete!")
                st.markdown(f"### Predicted Category: **{prediction}**")
                st.info("Confidence scores are not available for this model type.")
    else:
        st.warning("Please paste some resume text before classifying.")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This resume classifier uses a logistic regression model to automatically categorize resumes into job domains.

    **How it works:**
    1. Paste resume text into the text box (formatted or non-formatted will work)
    2. Click the "Classify Resume" button
    3. View the predicted job category and confidence scores

    **Categories include:**
    - Accountant
    - Advocate
    - Agriculture
    - Apparel
    - Arts
    - Automobile
    - Aviation
    - Banking
    - BPO
    - Business Development
    - Chef
    - Construction
    - Consultant
    - Designer
    - Digital Media
    - Engineering
    - Finance
    - Fitness
    - Healthcare
    - HR
    - Information Technology
    - Public Relations
    - Sales
    - Teacher
    """)

    st.header("Tips")
    st.markdown("""
    - Include ALL resume text for best results
    - The model analyzes skills, experience, and keywords
    - Longer, more detailed resumes typically produce better predictions
    """)

    st.markdown("---")
    st.caption("Resume Classifier: FA25: INTRO TO NLP FOR DATA SCIENCE: 9370 Final Project")
