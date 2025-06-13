import streamlit as st

#DASHBOARD

st.title('Stock Market Forecasting - BBNI.JK')
st.write('Dikerjakan untuk keperluan tugas project akhir UAS mata kuliah Machine Learning')
st.markdown("## Project Overview")

with st.expander(" Problem Statement"):
    st.markdown("""
    1. Metode tradisional seperti regresi atau analisis teknikal belum cukup efektif untuk menangkap pola harga saham yang kompleks.  
    2. Investor dan analis membutuhkan alat bantu yang mampu memberikan prediksi harga saham jangka pendek dengan lebih akurat. 
    """)

with st.expander(" Goals"):
    st.markdown("""
    1. Mengembangkan aplikasi web untuk memprediksi harga saham BBNI dalam jangka pendek.  
    2. Menerapkan 2 model dalam proses forecasting.  
    3. Menampilkan performa model dengan metrik MAE (Mean Absolute Error).  
    4. Mencatat hasil prediksi ke dalam file log untuk pelacakan histori.  
    5. Menyajikan hasil prediksi dalam bentuk tabel, grafik, dan statistik ringkas.
    6. Memberikan rekomendasi trading berdasarkan hasil prediksi.
    """)

with st.expander(" Solution Achieved"):
    st.markdown("""
    - Aplikasi Streamlit berhasil dikembangkan dan dapat menarik data historis saham BBNI.  
    - Model RNN dan LSTM berhasil diterapkan untuk melakukan prediksi harga saham.  
    - Hasil prediksi dapat divisualisasikan serta dievaluasi dengan MAE.  
    - Log prediksi disimpan ke supabase.  
    - Fitur histori log dan visualisasi interaktif tersedia di aplikasi.
    - Memberikan rekomendasi trading berdasarkan hasil prediksi.
    """)

st.markdown("### Sumber Terkait")
st.markdown(
    """
    - [Data Saham BBNI di Yahoo Finance](https://finance.yahoo.com/quote/BBNI.JK)
    - [Berita & Informasi Saham di Bursa Efek Indonesia (IDX)](https://www.idx.co.id)
    """
)