from ai_detector import predict_label, predict_proba

text = input("Paste a paragraph: ").strip()

label = predict_label(text)
score = predict_proba(text)

print("\nPrediction:", "AI" if label == 1 else "Human")
print("AI probability:", round(score, 4))
