from ai_detector import predict_proba

THRESHOLD_AI = 0.80   # tune this later using validation
THRESHOLD_HUMAN = 0.20

text = input("Paste a paragraph: ").strip()
p_ai = predict_proba(text)

if p_ai >= THRESHOLD_AI:
    label = "AI"
elif p_ai <= THRESHOLD_HUMAN:
    label = "Human"
else:
    label = "Uncertain"

print("\nPrediction:", label)
print("AI probability:", round(p_ai, 4))
