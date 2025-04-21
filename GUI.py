import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from customtkinter import *

# Load and prepare dataset
column_names = ['Date','Open','High','Low','Close','Volume',
                'Dividends','Stock Splits','Brand_Name','Ticker',
                'Industry_Tag','Country','Capital Gains']

dataset = pd.read_csv('World-Stock-Prices-Dataset.csv', names=column_names, header=0, na_values='?')
dataset['Date'] = pd.to_datetime(dataset['Date'], utc=True).dt.tz_localize(None)
dataset = dataset.dropna(subset=['Close'])
dataset['Year'] = dataset['Date'].dt.year
dataset['Month'] = dataset['Date'].dt.month
dataset['Day'] = dataset['Date'].dt.day

feature_columns = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Year', 'Month', 'Day']

scaler = StandardScaler()
scaler.fit(dataset[feature_columns])

label_encoder = LabelEncoder()
dataset['Ticker'] = dataset['Ticker'].astype(str)
label_encoder.fit(dataset['Ticker'])

model_dir = "model_files"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
model_paths = {name: os.path.join(model_dir, name) for name in model_files}

app = CTk()
app.geometry("400x450")
app.title("Stock Price Predictor")

def predict_price():
    user_ticker = ticker.get().strip().upper()
    selected_model_name = model_dropdown.get()
    
    if selected_model_name not in model_paths:
        result_label.configure(text="Please select a model.")
        return

    if user_ticker not in label_encoder.classes_:
        result_label.configure(text="Ticker not found!")
        return

    try:
        model = tf.keras.models.load_model(model_paths[selected_model_name])
    except Exception as e:
        result_label.configure(text=f"Error loading model:\n{e}")
        return

    ticker_data = dataset[dataset['Ticker'] == user_ticker].sort_values('Date')
    if ticker_data.empty:
        result_label.configure(text="No data for ticker.")
        return

    latest_row = ticker_data.iloc[-1]
    numerical_features = latest_row[feature_columns].values.reshape(1, -1)
    scaled_features = scaler.transform(numerical_features)
    reshaped = scaled_features.reshape(1, 1, len(feature_columns))  

    ticker_encoded = label_encoder.transform([user_ticker])

    prediction = model.predict({'num_input': reshaped, 'ticker_input': ticker_encoded}, verbose=0)
    predicted_price = prediction[0][0]

    result_label.configure(text=f"Predicted Close: ${predicted_price:.2f}")

ticker = CTkEntry(app, placeholder_text="Ticker")
ticker.pack(pady=20)

model_dropdown = CTkOptionMenu(app, values=model_files)
model_dropdown.set("Select Model")
model_dropdown.pack(pady=10)

generate_btn = CTkButton(app, text="Generate", command=predict_price)
generate_btn.pack(pady=10)

result_label = CTkLabel(app, text="")
result_label.pack(pady=20)

app.mainloop()
