import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from LSTM_impl import LSTM  
import matplotlib.pyplot as plt
import seaborn as sns


column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Brand_Name', 'Ticker',
                'Industry_Tag', 'Country', 'Capital Gains']

dataset = pd.read_csv('World-Stock-Prices-Dataset.csv', names=column_names, header=0, na_values='?')
dataset['Date'] = pd.to_datetime(dataset['Date'], utc=True).dt.tz_localize(None)
dataset = dataset.dropna(subset=['Close'])

dataset['Year'] = dataset['Date'].dt.year
dataset['Month'] = dataset['Date'].dt.month
dataset['Day'] = dataset['Date'].dt.day

dataset.sort_values(by=["Ticker", "Date"], inplace=True)

sequence_length = 10
features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Year', 'Month', 'Day']
target_column = 'Close'

sequences = []
targets = []

for ticker in dataset['Ticker'].unique():
    ticker_df = dataset[dataset['Ticker'] == ticker]
    X = ticker_df[features].values
    y = ticker_df[target_column].values

    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i+sequence_length])
        targets.append(y[i+sequence_length])

X = np.array(sequences)
y = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_shape = X_train.shape
X_test_shape = X_test.shape

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train_shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test_shape)

def evaluate_model(model, X_test, y_test):
    test_loss = 0
    for i in range(0, len(X_test), model.batch_size):
        x_batch = X_test[i:i+model.batch_size]
        y_batch = y_test[i:i+model.batch_size].reshape(-1, 1)
        loss = model.eval_step(x_batch, y_batch)
        test_loss += loss
    return test_loss / (len(X_test) / model.batch_size)

def train_model(model, X_train, y_train, X_test, y_test, epochs=1, initial_lr=0.001, lr_decay=0.95):
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        lr = initial_lr * (lr_decay ** epoch)

        for i in range(0, len(X_train), model.batch_size):
            idx = perm[i:i+model.batch_size]
            x_batch = X_train[idx]
            y_batch = y_train[idx].reshape(-1, 1)
            loss = model.train_step(x_batch, y_batch, lr=lr)
            epoch_loss += loss

        val_loss = evaluate_model(model, X_test, y_test)
        train_loss = epoch_loss / (len(X_train) / model.batch_size)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            model.save("best_model.keras")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

def main():
    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 2
    batch_size = 64

    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size
    )

    train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    final_loss = evaluate_model(model, X_test, y_test)
    print(f"Final Test Loss: {final_loss:.4f}")
    model.save("final_model.keras")

if __name__ == "__main__":
    main()