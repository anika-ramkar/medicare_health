import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Pima Indian Diabetes Dataset
data = pd.read_csv("diabetes.csv")

# Count the number of individuals with and without diabetes
diabetes_count = data['Outcome'].sum()
non_diabetes_count = len(data) - diabetes_count

print("Number of individuals with diabetes:", diabetes_count)
print("Number of individuals without diabetes:", non_diabetes_count)

# Split data into features (X) and target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)