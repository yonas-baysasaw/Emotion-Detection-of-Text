import joblib
import neattext.functions as nfx
import sys

# 1. Define Text Cleaning Functions (Copied from the original notebook)

def remove_stopwords(text):
    """Removes stopwords from the given text."""
    return nfx.remove_stopwords(text)

def remove_special_characters(text):
    """Removes special characters from the given text."""
    return nfx.remove_special_characters(text)

def remove_punctuations(text):
    """Removes punctuations from the given text."""
    return nfx.remove_punctuations(text)

#  2. Load the Trained Model and CountVectorizer
try:
    model = joblib.load('emotion_detector_model.joblib')
    print("Emotion detection model loaded successfully.")

    cv = joblib.load('count_vectorizer.joblib')
    print("CountVectorizer loaded successfully.")

except FileNotFoundError:
    print("Error: 'emotion_detector_model.joblib' or 'count_vectorizer.joblib' not found.")
    sys.exit(1)

# 3. Emotion Prediction Function

def predict_emotion(text):
    
    clean_text = remove_stopwords(text)
    clean_text = remove_special_characters(clean_text)
    clean_text = remove_punctuations(clean_text)

    vectorized_text = cv.transform([clean_text])

    # Predict the emotion
    prediction = model.predict(vectorized_text)[0]

    # Get the maximum probability for the predicted class
    prediction_proba = model.predict_proba(vectorized_text).max()

    return prediction, prediction_proba

# 4. Main Loop for User Input

if __name__ == "__main__":
    print("\n--- Emotion Detection from Text ---")
    print("Enter text to detect its emotion (type 'exit' to quit).")

    while True:
        user_input = input("\nEnter your text: ")

        if user_input.lower() == 'exit':
            print("Exiting emotion detection tool. Goodbye!")
            break

        if not user_input.strip():
            print("Please enter some text.")
            continue

        # Get prediction
        emotion, probability = predict_emotion(user_input)

        # Display results
        print(f"Predicted Emotion: {emotion.upper()}")
        print(f"Confidence: {probability:.2f}")

