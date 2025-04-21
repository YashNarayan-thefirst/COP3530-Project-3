import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from customtkinter import *
import webbrowser
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# Resolves an annoying multi-layer graph bug
current_canvas = None

def open_marketstack_signup():
    webbrowser.open("https://marketstack.com/signup")

def clear_previous_plot():
    global current_canvas
    if current_canvas:
        current_canvas.get_tk_widget().destroy()
        current_canvas = None
    plt.close('all')

try:
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
    
except Exception as e:
    print(f"Error initializing preprocessing: {e}")
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Year', 'Month', 'Day']
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

app = CTk()
app.title("Stock Price Predictor")
app.geometry("1200x800")

# Model loading
model_paths = {}
model_dir = "model_files"
supported_extensions = {'.keras', '.h5', '.tf'}

if os.path.isdir(model_dir):
    for filename in os.listdir(model_dir):
        filepath = os.path.join(model_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                model_name = os.path.splitext(filename)[0]
                model_paths[model_name] = filepath
else:
    print(f"Warning: Model directory '{model_dir}' not found.")

def predict_price():
    global current_canvas
    clear_previous_plot()
    
    user_ticker = ticker.get().strip().upper()
    selected_model_name = model_dropdown.get()
    api_key = api_key_entry.get().strip()

    if not api_key:
        result_label.configure(text="Please enter your MarketStack API key.", text_color="red")
        return

    if not user_ticker:
        result_label.configure(text="Please enter a stock ticker.", text_color="red")
        return

    if not model_paths:
        result_label.configure(text="No models found in model_files directory.", text_color="red")
        return

    if selected_model_name not in model_paths:
        result_label.configure(text="Please select a valid model.", text_color="red")
        return

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    url = "http://api.marketstack.com/v1/eod"
    params = {
        'access_key': api_key,
        'symbols': user_ticker,
        'date_from': start_date.strftime('%Y-%m-%d'),
        'date_to': end_date.strftime('%Y-%m-%d'),
        'limit': 1000
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        result_label.configure(text=f"Error fetching data: {str(e)}", text_color="red")
        return

    if 'data' not in data or not data['data']:
        result_label.configure(text="No data found for the given ticker.", text_color="red")
        return

    data_sorted = sorted(data['data'], key=lambda x: x['date'])
    closes = [entry['close'] for entry in data_sorted]
    dates = [entry['date'][:10] for entry in data_sorted]
    latest_close = closes[-1]  # Actual latest closing price

    # Prepare features
    latest_entry = data_sorted[-1]
    latest_features = np.array([
        latest_entry['open'],
        latest_entry['high'],
        latest_entry['low'],
        latest_entry['volume'],
        0,  # Dividends
        0,  # Stock Splits
        end_date.year,
        end_date.month,
        end_date.day
    ]).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(latest_features)
    reshaped = scaled_features.reshape(1, 1, -1)
    
    # Encode ticker
    if hasattr(label_encoder, 'classes_') and user_ticker in label_encoder.classes_:
        ticker_encoded = label_encoder.transform([user_ticker])
    else:
        ticker_encoded = np.array([hash(user_ticker) % 1000])

    # Make prediction
    try:
        model = tf.keras.models.load_model(model_paths[selected_model_name])
        prediction = model.predict({'num_input': reshaped, 'ticker_input': ticker_encoded}, verbose=0)
        predicted_price = prediction[0][0]
    
        error_percent = abs((predicted_price - latest_close) / latest_close) * 100
        
    except Exception as e:
        result_label.configure(text=f"Prediction error: {str(e)}", text_color="red")
        return

    result_text = (f"Actual Close: ${latest_close:.2f}\n"
                   f"Predicted Close: ${predicted_price:.2f}\n"
                   f"Error: {error_percent:.2f}%")
    result_label.configure(text=result_text, text_color="white")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, closes, label='Historical Prices', linewidth=2)
    ax.plot(dates[-1], latest_close, 'go', markersize=8, label='Actual Price')
    ax.plot(dates[-1], predicted_price, 'ro', markersize=8, label='Predicted Price')
    ax.set_title(f"{user_ticker} Price Prediction", fontsize=14, pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    current_canvas = FigureCanvasTkAgg(fig, master=app)
    current_canvas.draw()
    current_canvas.get_tk_widget().pack(pady=20)

frame = CTkFrame(app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

CTkLabel(frame, text="MarketStack API Key:").pack()
api_key_entry = CTkEntry(frame, width=300)
api_key_entry.pack(pady=5)

CTkLabel(frame, text="Stock Ticker:").pack()
ticker = CTkEntry(frame, placeholder_text="e.g., AAPL")
ticker.pack()

CTkLabel(frame, text="Select Model:").pack(pady=5)
model_dropdown = CTkComboBox(frame, values=sorted(model_paths.keys()))
model_dropdown.pack()

predict_btn = CTkButton(frame, text="Predict Price", command=predict_price)
predict_btn.pack(pady=10)

result_label = CTkLabel(frame, text="", font=("Arial", 14))
result_label.pack()

signup_link = CTkButton(app, text="Get MarketStack API Key", command=open_marketstack_signup)
signup_link.pack(pady=10)

app.mainloop()
