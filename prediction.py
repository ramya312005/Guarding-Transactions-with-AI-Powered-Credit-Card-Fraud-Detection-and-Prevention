from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('C:/Users/USER/OneDrive/文档/dataset.csv')  # Replace with your actual file path
print(df.columns)
# Example: If your target column is named 'Price'
X = df.drop('Class', axis=1)

# Define features and target  # Replace 'target_column' with your actual target column name
y = df['Class']
target = 'Class'  # Replace with your actual column name

if target in df.columns:
    X = df.drop(target, axis=1)
else:
    print(f"'{target}' not found in DataFrame")



# Assuming X and y are your features and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
def load_data():
    # Load your dataset
    # Split into features and target
    # Split into training and testing sets
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()


model = LogisticRegression()
model.fit(X_train, y_train)

import pickle
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

# Save the model
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("new model.pickle file created successfully")


import pickle
import numpy as np

# Function to load the trained model from a file
def load_model(model_path="model.pkl"):
   with open("credit_model.pkl", "rb") as f:
       model = pickle.load(f)
model = load_model()
print("✅ Model loaded successfully")

# Function to predict fraud on a given transaction
def predict_fraud(model, transaction_data):
    try:
        # Ensure the input data is in the correct shape (2D array)
        input_data_reshaped = np.array(transaction_data).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_data_reshaped)

        # Return the prediction result
        return prediction[0]  # Returning single value prediction
    except Exception as e:
        print(f"❌ Error in making prediction: {e}")
        return None
    

import pandas as pd
import numpy as np
import pickle

# Load the trained model
def load_model(model_path="credit_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()  # Load your pre-trained model

# Function to predict fraud on a given transaction
def predict_fraud(model, transaction_data):
    try:
        # Ensure the input data is in the correct shape (2D array)
        input_data_reshaped = np.array(transaction_data).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_data_reshaped)

        # Return the prediction result
        return prediction[0]  # Returning single value prediction
    except Exception as e:
        print(f"❌ Error in making prediction: {e}")
        return None

# --- Process transactions from a CSV file ---
def process_transactions_from_csv(file_path="C:/Users/USER/OneDrive/文档/dataset.csv"):
    try:
        # Load the new transactions
        new_data = pd.read_csv(file_path)

        # Ensure no target column is included (e.g., 'Class')
        if 'Class' in new_data.columns:
            new_data = new_data.drop(columns=['Class'])

        # Predict for each row in the CSV
        for index, row in new_data.iterrows():
            prediction = predict_fraud(model, row.values)
            print(f"Transaction {index + 1}: {'🚨 Fraud' if prediction == 1 else '✅ Approved'}")
    except Exception as e:
        print(f"❌ Error in processing CSV file: {e}")

# Example usage: Processing new transactions from a CSV file
process_transactions_from_csv("C:/Users/USER/OneDrive/文档/dataset.csv")




