import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import joblib
import time
import os
import requests
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

SUPABASE_URL = "https://lkjoktgpwolixrsfomsh.supabase.co"  
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxram9rdGdwd29saXhyc2ZvbXNoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NjM4NjEsImV4cCI6MjA2NTMzOTg2MX0.HMc1cBnc0ZN2J0AgzEn4I9h-oP08OVPzDvJIsgGWb68"  
SUPABASE_TABLE = "prediction_log"
def load_log_from_supabase():
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Gagal mengambil data log: {response.text}")
        return pd.DataFrame()


def log_predictions(log_date, model_name, forecast_df, input_sequences):
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    for i, row in forecast_df.iterrows():
        input_str = '|'.join(map(str, input_sequences[i].flatten()))
        data = {
            "log_date": str(log_date),
            "model": model_name,
            "predicted_date": str(row['Date'].date()),
            "input_sequence": input_str,
            "predicted_value": float(row['Predicted'])
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 201:
            print(f"Gagal kirim log baris ke-{i}: {response.text}")
        else:
            print(f"Berhasil kirim log baris ke-{i}")

st.title("Whats'up Traders Fokus cuan Todayy!!")
st.info ("⚠️ Disclaimer Gunakan ini sebagai referensi")



#SIDEEEE
st.sidebar.subheader('INPUT DULU BRODYY')
stock = 'BBNI.JK'
start_date = st.sidebar.date_input('Pilih Tanggal Mulai', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Pilih Tanggal Akhir', pd.to_datetime('today'))
n_days = st.sidebar.number_input("Jumlah Hari ke Depan untuk Forecasting",10, 120)
model_choice = st.sidebar.selectbox("Pilih Model", ['RNN', 'LSTM'])
show_mae = st.sidebar.checkbox("Tampilkan MAE")
button = st.sidebar.button("SHOW NOW!!")
see_log = st.sidebar.button("Lihat Log Prediksi")

#SHOWW LOG
if see_log:
    st.subheader("Log Prediksi")
    log_df = load_log_from_supabase()
    if not log_df.empty:
        st.dataframe(log_df)
    else:
        st.warning("Tidak ada data log yang tersedia.")

#SHOWRESULT
if button:
    placeholder = st.empty()
    placeholder.progress(50, "loading....")
    time.sleep(1)
    placeholder.progress(100, "loading....")
    time.sleep(1)

    with placeholder.container():
        time.sleep(1)
        st.markdown("Mengunduh Data...")
        time.sleep(2)
        st.markdown("Masih Mengunduhh....")
        time.sleep(2)
        st.markdown("Selesai Mengunduh..")
        time.sleep(2)

    placeholder.title("enjoyy!! Tipis Tipis..")
    time.sleep(2)

    placeholder.empty()
    try:
        df = yf.download(stock, start=start_date, end=end_date)
        df.reset_index(inplace=True)
    except Exception as e:
        st.error(f"Gagal mengunduh data: {e}")

    st.subheader('Data Historis Saham')
    st.write(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot.line(y="Close", x="Date", ax=ax)
    st.pyplot(fig)
    

    model_file = f"model_{model_choice.lower()}.h5"
    scaler_file = f"scaler_{model_choice.lower()}.save"

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        st.error(f"Model atau scaler untuk {model_choice} belum tersedia.")
    else:
        with st.spinner(f"Memuat model dan melakukan prediksi ({model_choice})..."):
            model = load_model(model_file, compile=False)
            scaler = joblib.load(scaler_file)

            lookback = 10
            close_prices = df['Close'].values.reshape(-1, 1)
            scaled_prices = scaler.transform(close_prices)

            input_seq = scaled_prices[-lookback:]
            preds_scaled = []
            input_seqs_log = []

            for _ in range(n_days):
                pred = model.predict(input_seq.reshape(1, lookback, 1), verbose=0)
                preds_scaled.append(pred[0])
                input_seqs_log.append(input_seq.copy())
                input_seq = np.append(input_seq[1:], pred, axis=0)

            preds = scaler.inverse_transform(np.array(preds_scaled))
            last_date = df['Date'].iloc[-1]
            forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted': preds.flatten()
            })

            
            log_predictions(datetime.today().date(), model_choice, forecast_df, input_seqs_log)
            st.success("Prediksi berhasil dilakukan dan disimpan ke log.")

            
            st.subheader("Ringkasan Statistik Prediksi")
            st.metric("Rata-rata", f"{float(preds.mean()):,.2f}")
            st.metric("Tertinggi", f"{float(preds.max()):,.2f}")
            st.metric("Terendah", f"{float(preds.min()):,.2f}")

            st.subheader("Data Prediksi")
            st.write(forecast_df)

            st.subheader(f'Visualisasi Prediksi ({model_choice})')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Date'], df['Close'], label='Actual')
            ax.plot(forecast_df['Date'], forecast_df['Predicted'], label=f'Forecast - {model_choice}', linestyle='--', marker='o')
            ax.legend()
            st.pyplot(fig)

            if show_mae:
                if len(df) >= lookback + n_days:
                    real_values = close_prices[-n_days:].flatten()
                    predicted_values = preds.flatten()
                    mae = mean_absolute_error(real_values, predicted_values)
                    st.success(f"MAE antara prediksi dan data aktual: {mae:.4f}")
                else:
                    st.warning("Data historis tidak cukup untuk menghitung MAE.")

            #Rekomendasi
            st.subheader(" Rekomendasi Trading Dari Akang")

            try:
                start_price = forecast_df['Predicted'].iloc[0]
                end_price = forecast_df['Predicted'].iloc[-1]

                if end_price > start_price * 1.01:
                    recommendation = "Beli"
                    advice = "Harga diprediksi naik, pertimbangkan untuk membeli saham."
                elif end_price < start_price * 0.99:
                    recommendation = "Jual"
                    advice = "Harga diprediksi turun, pertimbangkan untuk menjual saham."
                else:
                    recommendation = "Tahan"
                    advice = "Pergerakan harga cenderung stabil, disarankan untuk menahan posisi."

                st.success(f"**Rekomendasi: {recommendation}**")
                st.info(advice)

            except Exception as e:
                st.error(f"Gagal menghasilkan rekomendasi trading: {e}")

