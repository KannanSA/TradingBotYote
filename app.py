# /* Copyright (C) Kannan Sekar Annu Radha - All Rights Reserved
#  * Unauthorized copying of this file, via any medium is strictly prohibited
#  * Proprietary and confidential
#  * Written by Kannan Sekar Annu Radha <kannansekara@gmail.com>, April 2024
#  */ 
from flask import Flask, render_template, request, redirect, url_for, session
from binance.client import Client
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf
import time
import json
import bcrypt

# Initialize Flask app
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# # Initialize Binance client
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'

client = Client(API_KEY, API_SECRET)

# Dummy user data
users = {
    'user1': bcrypt.hashpw(b'password1', bcrypt.gensalt()),
    'user2': bcrypt.hashpw(b'password2', bcrypt.gensalt())
}

# Dummy API data
apis = {
    'key1': bcrypt.hashpw(b'secret1234', bcrypt.gensalt()),
    'key2': bcrypt.hashpw(b'secret5678', bcrypt.gensalt())
}

def get_eth_data(interval='1h', limit=100):
    try:
        klines = client.get_klines(symbol='ETHUSDT', interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching ETH data: {e}")
        return None

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1,1))
    return scaled_data, scaler

def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)

def predict_next_price(model, last_60_prices):
    prediction = model.predict(last_60_prices.reshape(1,60,1))
    return prediction

def generate_lstm_predictions(model, eth_data_scaled, scaler):
    # Use the model to predict future prices based on the last 60 prices
    last_60_prices = eth_data_scaled[-60:]
    predictions = []
    for _ in range(10):  # Predict 10 future prices
        prediction = predict_next_price(model, last_60_prices)
        predictions.append(scaler.inverse_transform(prediction)[0][0])
        last_60_prices = np.append(last_60_prices[1:], prediction, axis=0)
    return predictions

def convert_to_python_types(predictions):
    return [float(prediction) for prediction in predictions]

# Routes

@app.route('/')
def index():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    else:
        return redirect(url_for('trading'))

@app.route('/trading')
def trading():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    eth_data = get_eth_data()
    if eth_data is None:
        return render_template('error.html', message="Error fetching ETH data. Please try again later.")

    eth_data_scaled, scaler = preprocess_data(eth_data)

    model = create_model()
    X, y = create_dataset(eth_data_scaled)
    train_model(model, X, y)

    last_60_prices = eth_data_scaled[-60:]
    prediction = predict_next_price(model, last_60_prices)
    next_price = scaler.inverse_transform(prediction)
    lstm_predictions = generate_lstm_predictions(model, eth_data_scaled, scaler)

    return render_template('trading.html', eth_data=eth_data.to_html(), next_price=next_price[0][0], lstm_predictions=lstm_predictions)

@app.route('/get-predictions')
def get_predictions():
    eth_data = get_eth_data()
    if eth_data is None:
        return {'predictions': []}

    eth_data_scaled, scaler = preprocess_data(eth_data)

    model = create_model()
    X, y = create_dataset(eth_data_scaled)
    train_model(model, X, y)

    lstm_predictions = generate_lstm_predictions(model, eth_data_scaled, scaler)

    # Convert float32 values to Python native types
    lstm_predictions = convert_to_python_types(lstm_predictions)

    return {'predictions': lstm_predictions}

@app.route('/enter-api-keys', methods=['GET', 'POST'])
def enter_api_keys():
    if request.method == 'POST':
        api_key = request.form['api_key']
        api_secret = request.form['api_secret']
        session['api_key'] = api_key
        session['api_secret'] = api_secret
        return redirect(url_for('trading'))
    return render_template('enter_api_keys.html')

@app.route('/manual-buy')
def manual_buy():
    # Implement the logic for manual buy action
    # For example, you can place a buy order using the Binance API
    return {'message': 'Manual buy action executed successfully'}

@app.route('/manual-sell')
def manual_sell():
    # Implement the logic for manual sell action
    # For example, you can place a sell order using the Binance API
    return {'message': 'Manual sell action executed successfully'}

# User authentication routes

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]):
            session['username'] = username
            session['logged_in'] = True
            return redirect(url_for('trading'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/create-account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[username] = hashed_password
        return redirect(url_for('login'))
    return render_template('create_account.html')

if __name__ == '__main__':
    app.run(debug=True)