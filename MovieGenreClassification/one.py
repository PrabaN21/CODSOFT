import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# List of Genres from the training dataset
genre_list = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'family',
              'fantasy', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance',
              'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western']

# Load the training dataset
try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('train_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise

# Preprocessing training data
X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Split data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model pipeline
model = Pipeline([
    ('tfidfvectorizer', tfidf_vectorizer),
    ('multioutputclassifier', MultiOutputClassifier(LinearSVC()))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'multioutputclassifier__estimator__C': [1, 10],
    'multioutputclassifier__estimator__penalty': ['l2'],
    'tfidfvectorizer__max_features': [10000, 15000],
    'tfidfvectorizer__ngram_range': [(1, 2)]
}

# Use KFold for cross-validation instead of StratifiedKFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=kf, verbose=2, n_jobs=-1, scoring='accuracy')

# Fit the model using grid search
grid_search.fit(X_train_split, y_train_split)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the validation set
predicted_genres = best_model.predict(X_val_split)

# Create a DataFrame with movie names and predicted genres
test_results = pd.DataFrame({
    'MOVIE_NAME': train_data.iloc[X_val_split.index]['MOVIE_NAME'],  # Use X_val_split.index to get the correct movie names
    'PREDICTED_GENRES': mlb.inverse_transform(predicted_genres)
})

# Save the predicted genres to a CSV file
test_results.to_csv('predicted_movie_genres.csv', index=False)

# Print the evaluation metrics
accuracy = accuracy_score(y_val_split, predicted_genres)
precision = precision_score(y_val_split, predicted_genres, average='micro')
recall = recall_score(y_val_split, predicted_genres, average='micro')
f1 = f1_score(y_val_split, predicted_genres, average='micro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Save the model evaluation metrics
with open('model_evaluation.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1-score: {f1:.2f}\n")
