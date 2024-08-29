
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from datetime import date, timedelta
# import streamlit as st

# # Load the pre-trained LSTM model
# model = tf.keras.models.load_model('keras_model.h5')

# # Set up the start and end dates for fetching stock data
# START = "2010-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")

# # Define a function to load stock data from Yahoo Finance
# def load_data(ticker):
#     data = yf.download(ticker, START, TODAY)
#     data.reset_index(inplace=True)
#     return data

# # Streamlit app
# st.title('Stock Price Prediction')

# # User input for stock ticker
# ticker = st.text_input('Enter Stock Ticker:', '')

# if ticker:
#     # Fetch and preprocess data
#     data = load_data(ticker)
#     df = data.copy()
#     df = df.drop(['Date', 'Adj Close'], axis=1)

#     # Prepare the data for prediction
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     last_100_days = df['Close'].values[-100:].reshape(-1, 1)
#     last_100_days_scaled = scaler.fit_transform(last_100_days)
#     input_data = last_100_days_scaled.reshape((1, last_100_days_scaled.shape[0], 1))

#     # Predict the next 10 days
#     next_10_days_predictions = []

#     for day in range(10):
#         predicted_price_scaled = model.predict(input_data)
#         predicted_price = scaler.inverse_transform(predicted_price_scaled)
#         next_10_days_predictions.append(predicted_price[0][0])
#         # Append predicted price to the input data
#         predicted_price_scaled = predicted_price_scaled.reshape((1, 1, 1))  # Ensure correct shape
#         input_data = np.concatenate((input_data[:, 1:, :], predicted_price_scaled), axis=1)

#     # Generate the next 10 days' dates
#     last_date = data['Date'].iloc[-1]
#     next_10_days_dates = [last_date + timedelta(days=i) for i in range(1, 11)]

#     # Create a DataFrame to display the predicted prices with dates
#     predicted_next_10_days_df = pd.DataFrame({'Date': next_10_days_dates, 'Predicted_Price': next_10_days_predictions})

#     # Display results
#     st.write(predicted_next_10_days_df)

#     # Plot results
#     st.line_chart(predicted_next_10_days_df.set_index('Date'))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('keras_model.h5')

# Define start and end dates
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Function to load stock data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Streamlit app layout
st.title('Stock Price Prediction for Next 10 Days')

ticker = st.text_input('Enter Stock Ticker Symbol', 'AAPL')

if st.button('Predict'):
    if ticker:
        # Fetch and preprocess data
        data = load_data(ticker)
        df = data.copy()
        df = df.drop(['Date', 'Adj Close'], axis=1)
        
        # Initialize scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Prepare last 100 days of data
        last_100_days = df['Close'].values[-100:].reshape(-1, 1)
        last_100_days_scaled = scaler.fit_transform(last_100_days)
        input_data = last_100_days_scaled.reshape((1, last_100_days_scaled.shape[0], 1))

        # Predict the next 10 days
        next_10_days_predictions = []
        for day in range(10):
            predicted_price_scaled = model.predict(input_data)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)
            next_10_days_predictions.append(predicted_price[0][0])
            predicted_price_scaled = predicted_price_scaled.reshape((1, 1, 1))
            input_data = np.concatenate((input_data[:, 1:, :], predicted_price_scaled), axis=1)

        # Generate dates for the next 10 days
        last_date = data['Date'].iloc[-1]
        next_10_days_dates = [last_date + timedelta(days=i) for i in range(1, 11)]

        # Create a DataFrame to display the predicted prices with dates
        predicted_next_10_days_df = pd.DataFrame({
            'Date': next_10_days_dates,
            'Predicted_Price': next_10_days_predictions
        })

        # Display the predictions
        st.write(predicted_next_10_days_df)

        # Plot the predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(predicted_next_10_days_df['Date'], predicted_next_10_days_df['Predicted_Price'], marker='o', linestyle='-', color='r', label='Predicted Price')
        plt.title('Predicted Stock Prices for the Next 10 Days')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price (INR)')
        plt.grid(True)
        plt.legend()
        st.pyplot()
    else:
        st.error('Please enter a stock ticker symbol.')
