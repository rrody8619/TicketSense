import joblib
from ml import clean_text

sentiment_model = joblib.load('models/sentiment_tfidf_lr_pipeline.pkl')

def analyse(text):
    cleaned = clean_text(text)
    result = sentiment_model.predict([cleaned])[0]
    return result