# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
dataset_path = r"C:\Users\praba\PycharmProjects\CodSoftSpamSMS\spam.csv"  # Use the raw string for proper path handling
data = pd.read_csv(dataset_path, encoding='latin-1')

# Step 3: Data preprocessing
data_subset = data[['v1', 'v2']]  # Select only the relevant columns
data_subset.columns = ['category', 'text_message']  # Rename columns for clarity

# Convert 'spam' to 1 and 'ham' to 0 for easier model interpretation
data_subset['category'] = data_subset['category'].apply(lambda x: 1 if x == 'spam' else 0)

# Step 4: Split the dataset into training and testing subsets
X_features = data_subset['text_message']
y_labels = data_subset['category']
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Step 5: Transform the text data into numerical form using TF-IDF
tfidf_transformer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)

# Step 6: Train the logistic regression model
spam_classifier = LogisticRegression()
spam_classifier.fit(X_train_tfidf, y_train)

# Step 7: Make predictions and evaluate the model
predictions = spam_classifier.predict(X_test_tfidf)

# Display evaluation metrics
print("Accuracy of the model:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Step 8: Save the model and vectorizer for future use (optional)
import joblib
joblib.dump(spam_classifier, "spam_classifier_model.pkl")
joblib.dump(tfidf_transformer, "tfidf_vectorizer.pkl")
