import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Wisconsin Breast Cancer Dataset
data = pd.read_csv("breast_cancer.csv")

# Count the number of individuals with and without cancer
cancer_count = data['diagnosis'].value_counts()['M']
non_cancer_count = data['diagnosis'].value_counts()['B']

print("Number of individuals with cancer:", cancer_count)
print("Number of individuals without cancer:", non_cancer_count)

# Split data into features (X) and target variable (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

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
