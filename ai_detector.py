import joblib

MODEL_PATH = "ai_detector_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

_model = joblib.load(MODEL_PATH)
_vectorizer = joblib.load(VECTORIZER_PATH)

def predict_label(text: str) -> int:
    """Returns 1 if AI-generated, 0 if Human."""
    vec = _vectorizer.transform([text])
    return int(_model.predict(vec)[0])

def predict_proba(text: str) -> float:
    """Returns probability of AI (class 1)."""
    vec = _vectorizer.transform([text])
    proba_ai = _model.predict_proba(vec)[0][1]
    return float(proba_ai)
