import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI Demand Forecasting Dashboard", layout="wide")
st.title("AI Demand Forecasting Dashboard")
st.write("Demand prediction for Store-Item combinations using LSTM")

# Upload dataset
uploaded_file = st.file_uploader("Upload Store Item Demand Dataset CSV", type=["csv"])
if not uploaded_file:
    st.warning("Please upload the 'train.csv' file from the Kaggle demand forecasting challenge dataset.")
    st.stop()
    
data = pd.read_csv(uploaded_file)
data['date'] = pd.to_datetime(data['date'])

# Show store and item filters
stores = sorted(data['store'].unique())
items = sorted(data['item'].unique())
selected_store = st.selectbox("Select Store", stores)
selected_item = st.selectbox("Select Item", items)

# Filter data for selected store and item
df = data[(data['store'] == selected_store) & (data['item'] == selected_item)][['date', 'sales']]
df = df.sort_values('date')
df.set_index('date', inplace=True)

# Show preview
st.subheader(f"Sales Data Preview (Store {selected_store} - Item {selected_item})")
st.line_chart(df['sales'])

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['sales']])

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window = st.slider("Sequence Window (days)", min_value=7, max_value=90, value=30)
epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20)
batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=16)

X, y = create_sequences(scaled_data, window)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
with st.spinner("Training LSTM model..."):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window, 1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    st.success("Model training complete!")

# Predict and visualize
predictions = model.predict(X_test)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted = scaler.inverse_transform(predictions)

st.subheader("Demand Forecasting Result")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(actual, label='Actual Sales')
ax.plot(predicted, label='Predicted Sales')
ax.set_xlabel('Time Index')
ax.set_ylabel('Sales')
ax.legend()
st.pyplot(fig)

# Download results
results = pd.DataFrame({
    "Actual Sales": actual.flatten(),
    "Predicted Sales": predicted.flatten()
})
st.download_button("Download Forecast Results", results.to_csv(index=False), "forecast_results.csv")

# Loss curve
st.subheader("Training Loss Curve")
fig2, ax2 = plt.subplots()
ax2.plot(hist.history['loss'], label='Training Loss')
ax2.plot(hist.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
st.pyplot(fig2)