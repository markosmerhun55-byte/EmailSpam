import joblib
import pandas as pd

# 1. Load the binary file back into RAM
pipeline = joblib.load("best_sms_spam_model.joblib")

# 2. Extract the vectorizer and classifier components
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']

# 3. See the learned vocabulary size
feature_names = vectorizer.get_feature_names_out()
print(f"Total learned words in vocabulary: {len(feature_names)}")

# 4. Get feature coefficients (weights) from Linear SVM
weights = classifier.coef_[0]

# 5. Create a readable DataFrame
word_importance = pd.DataFrame({
    'Word': feature_names,
    'Weight': weights
}).sort_values(by='Weight', ascending=False)

print("\n--- Top 10 Words that Trigger SPAM Classification ---")
print(word_importance.head(10).to_string(index=False))

print("\n--- Top 10 Words that Trigger HAM Classification ---")
print(word_importance.tail(10).to_string(index=False))