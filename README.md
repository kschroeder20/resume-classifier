# Resume Classifier

A machine learning-powered web application that automatically classifies resumes into job categories.

## Live Demo

[Try the app here!](#) *(Add your Streamlit Cloud URL after deployment)*

## Features

- 📄 Paste resume text directly into the app
- 🤖 ML-powered classification using TF-IDF and LinearSVC/LogisticRegression
- 📊 Confidence scores for all job categories
- 🎯 Supports 24+ job categories including:
  - Data Science, Software Engineering, HR
  - Marketing, Finance, Healthcare
  - And many more...

## How to Use

1. Copy resume text (plain text format works best)
2. Paste into the text area
3. Click "Classify Resume"
4. View the predicted job category and confidence scores

## Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Scikit-learn (TF-IDF + Linear SVC/Logistic Regression)
- **NLP**: NLTK for text preprocessing
- **Deployment**: Streamlit Community Cloud

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Model Performance

- Test Accuracy: ~71%
- Test F1 Score: ~70%
- Trained on 2,484 resumes across 24 job categories

## Team

Team: Allen Snyder, Kevin Schroeder, Si Liang

Created: 10/30/2025
