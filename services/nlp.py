import joblib
import os

from .ml import clean_text 

BASE_DIR = os.path.dirname(__file__)
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_tfidf_lr_pipeline.pkl')


sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)

def analyse(text):
    cleaned = clean_text(text)
    result = sentiment_model.predict([cleaned])[0]
    return result
