import joblib
import re
import string
import nltk
import os  # ضفنا دي عشان المسارات
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# تنزيل ملفات NLTK المهمة للسيرفر
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 

lemmatizer = WordNetLemmatizer()

# تحديد المسار التلقائي (عشان يشتغل على جهازك وعلى السيرفر)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# تحميل الموديلات بالمسار الصحيح والأسماء اللي رفعتيها
category_model = joblib.load(os.path.join(MODEL_DIR, 'category_tfidf_lr_pipeline.pkl'))
priority_model = joblib.load(os.path.join(MODEL_DIR, 'priority_tfidf_lr_pipeline.pkl'))
# ضيفي السطر ده لو هتحتاجي السينتمنت برضه
# sentiment_model = joblib.load(os.path.join(MODEL_DIR, 'sentiment_tfidf_lr_pipeline.pkl'))

def clean_text(text):
    if not isinstance(text, str): text = str(text) # تأمين لو المدخل مش نص
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def predict_ml(text):
    cleaned = clean_text(text)
    category = category_model.predict([cleaned])[0]
    priority = priority_model.predict([cleaned])[0]
    return category, priority