import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the dataset from the specified path
data_path = r"C:\Users\praba\PycharmProjects\CustomerChurnPrediction\Churn_Modelling.csv"
churn_data = pd.read_csv(data_path)

# Step 2: Clean the dataset by removing unnecessary columns
drop_columns = ['RowNumber', 'CustomerId', 'Surname']  # Drop columns that aren't needed for modeling
churn_data_cleaned = churn_data.drop(columns=drop_columns, axis=1)

# Step 3: One-hot encode categorical columns
categorical_columns = ['Geography', 'Gender']
encoded_data = pd.get_dummies(churn_data_cleaned, columns=categorical_columns, drop_first=True)

# Step 4: Separate features and target variable
X_features = encoded_data.drop('Exited', axis=1)  # Drop target variable from the features
y_target = encoded_data['Exited']  # Target variable for classification

# Step 5: Standardize numerical features to bring them to the same scale
scaler = StandardScaler()
X_scaled_features = scaler.fit_transform(X_features)

# Step 6: Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_features, y_target, test_size=0.3, random_state=42, stratify=y_target)

# Step 7: Train the Random Forest model with balanced class weights
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
random_forest_classifier.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = random_forest_classifier.predict(X_test)

# Step 9: Evaluate model performance using accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Step 10: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 11: Visualize feature importance using a bar plot
feature_importances_df = pd.DataFrame({
    'Feature': X_features.columns,
    'Importance': random_forest_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette='coolwarm')
plt.title("Feature Importance in Predicting Churn")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Step 12: Display evaluation metrics and confusion matrix
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
