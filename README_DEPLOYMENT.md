# Resume Classifier - Streamlit Deployment Guide

This guide will help you deploy the Resume Classifier application using Streamlit.

## Prerequisites

Before running the app, make sure you have:
1. Trained the model by running all cells in `Resume_Classifier.ipynb`
2. Generated the model files (`best_resume_classifier_model.pkl` and `preprocessing_utils.pkl`)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App Locally

1. Make sure you're in the `resume_classifier` directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The app will open in your default web browser (typically at `http://localhost:8501`)

## Using the App

1. Copy and paste resume text into the text area
2. Click the "Classify Resume" button
3. View the predicted job category and confidence scores

## Deployment to Production

### Option 1: Streamlit Community Cloud (Free)

1. Push your code to a GitHub repository
2. Make sure these files are in your repo:
   - `app.py`
   - `requirements.txt`
   - `best_resume_classifier_model.pkl`
   - `preprocessing_utils.pkl`

3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Sign in with GitHub
5. Click "New app" and select your repository
6. Set the main file path to `app.py`
7. Click "Deploy"

### Option 2: Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t resume-classifier .
docker run -p 8501:8501 resume-classifier
```

### Option 3: Cloud Platforms

Deploy to:
- **Heroku**: Use the Streamlit buildpack
- **AWS EC2**: Run the app on an EC2 instance
- **Google Cloud Run**: Deploy as a containerized app
- **Azure App Service**: Deploy directly from GitHub

## File Structure

```
resume_classifier/
├── app.py                              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── Resume_Classifier.ipynb             # Model training notebook
├── best_resume_classifier_model.pkl    # Trained model (generated)
├── preprocessing_utils.pkl             # Preprocessing functions (generated)
└── README_DEPLOYMENT.md                # This file
```

## Troubleshooting

**Model files not found error:**
- Make sure you've run the Jupyter notebook completely
- Check that `best_resume_classifier_model.pkl` and `preprocessing_utils.pkl` exist in the same directory as `app.py`

**NLTK stopwords error:**
- The app will automatically download stopwords on first run
- If issues persist, manually run: `python -c "import nltk; nltk.download('stopwords')"`

**Memory issues during deployment:**
- Consider using a smaller model or reducing `max_features` in TF-IDF
- Use cloud platforms with sufficient memory (at least 1GB RAM recommended)

## Security Considerations

For production deployment:
1. Add rate limiting to prevent abuse
2. Implement input validation and sanitization
3. Set up monitoring and logging
4. Use HTTPS for secure connections
5. Consider adding authentication if needed

## Support

For issues or questions, please refer to the project documentation or contact the development team.
