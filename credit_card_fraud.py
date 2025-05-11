import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/USER/OneDrive/Desktop/deployment machine learning/credit_model (3).pkl', 'rb'))

# Function for prediction
def fraud_detection(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape for single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict using model
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'Transaction is NOT fraudulent'
    else:
        return 'Transaction is FRAUDULENT!'


# Main function for UI
def main():
    st.title('Credit Card Fraud Detection Web App')

    # Collect user input - adjust fields based on your dataset features
    Time = st.text_input('Time (seconds since first transaction)')
    V1 = st.text_input('V1')
    V2 = st.text_input('V2')
    V3 = st.text_input('V3')
    V4 = st.text_input('V4')
    V5 = st.text_input('V5')
    Amount = st.text_input('Transaction Amount')

    # Result message
    result = ''
 # Button to predict
    if st.button('Check Transaction'):
        try:
            input_features = [Time, V1, V2, V3, V4, V5, Amount]
            result = fraud_detection(input_features)
        except:
            result = "Please enter valid numeric input values!"

    st.success(result)


if __name__ == '__main__':
 main()