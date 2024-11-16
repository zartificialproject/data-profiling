import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Data Profiling by Zartificial", page_icon="🔍")

# Judul Aplikasi
st.title("🔍 Data Profiling & Anomaly Detector")

# Instruksi Penggunaan
st.markdown("Unggah file data Anda dalam format **Excel** atau **CSV** untuk melakukan profiling data dan mendeteksi anomali.")
st.markdown("---")

# Unggah Data
uploaded_file = st.file_uploader("📂 Unggah file data (Excel atau CSV)", type=["xlsx", "csv"], key="data_uploader")
if uploaded_file:
    # Pesan keberhasilan unggah
    st.success("File berhasil diunggah! Memuat data...")

    # Membaca data sesuai format file
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)

    # Menangani kolom tanggal jika ada
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                # Mengonversi kolom yang berformat string tanggal menjadi datetime
                data[col] = pd.to_datetime(data[col], format='%d.%m.%Y', errors='coerce')
            except Exception as e:
                pass  # Abaikan jika kolom bukan tanggal
    
    # Menampilkan Ringkasan Data dalam Kolom
    st.markdown("### 📊 Ringkasan Data")
    st.write(data.describe())

    # Missing Values Section
    st.markdown("### ❓ Missing Values")
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.write(missing_values)
    else:
        st.info("Tidak ada missing values yang ditemukan dalam dataset.")

    # Visualisasi Tren untuk Variabel Terpilih
    st.markdown("### 📈 Visualisasi Tren Variabel")
    st.write("Pilih kolom numerik untuk melihat tren perubahan data dalam bentuk plot garis.")
    
    trend_col = st.selectbox("Pilih Kolom untuk Visualisasi Tren", data.select_dtypes(include=np.number).columns)
    if trend_col:
        st.write(f"Visualisasi tren untuk **{trend_col}**")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data[trend_col], marker='o', linestyle='-', color='b', label=trend_col)
        ax.set_title(f"Tren Variabel {trend_col}")
        ax.set_xlabel('Indeks')
        ax.set_ylabel(trend_col)
        ax.legend()
        st.pyplot(fig)

    # Korelasi Heatmap
    st.markdown("### 🔥 Korelasi Heatmap")
    st.write("Heatmap digunakan untuk melihat korelasi antar kolom numerik. Nilai yang lebih tinggi menunjukkan korelasi yang lebih kuat antara dua kolom.")
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Boxplot untuk Deteksi Anomali
    st.markdown("### 📉 Deteksi Anomali dengan Boxplot")
    st.write("Boxplot digunakan untuk mendeteksi anomali dalam distribusi data. Garis horizontal pada boxplot menunjukkan nilai median, kotak menunjukkan interquartile range (IQR), dan garis vertikal menunjukkan data yang berada di luar batas IQR yang dianggap sebagai outlier.")
    col = st.selectbox("Pilih Kolom untuk Boxplot", data.select_dtypes(include=np.number).columns)
    if col:
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)
        
    # Deteksi Anomali dengan Z-Score
    st.markdown("### 📊 Deteksi Anomali Lanjutan dengan Z-Score")
    z_threshold = st.slider("Pilih Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
    z_outliers = {}
    for col in data.select_dtypes(include=np.number).columns:
        z_scores = stats.zscore(data[col].dropna())
        outliers = np.where(np.abs(z_scores) > z_threshold)
        z_outliers[col] = outliers
        if len(outliers[0]) > 0:
            st.write(f"**Kolom {col}** memiliki {len(outliers[0])} outliers berdasarkan Z-Score.")
            outlier_data = data.iloc[outliers[0]]
            st.write("Data Outlier: ")
            st.write(outlier_data)

    

    # Summary Insight Section
    st.markdown("### 🧩 Summary Insight")
    
    # Flag untuk memeriksa apakah ada insight ditemukan
    insight_found = False
    insights = []
    
    for col in data.select_dtypes(include=np.number).columns:
        # Menampilkan informasi missing values
        if data[col].isnull().mean() > 0:
            insights.append(f"- Kolom **{col}** memiliki missing values sebesar {data[col].isnull().mean() * 100:.2f}%")
            insight_found = True
        # Menampilkan informasi outlier berdasarkan Z-score
        z_scores = stats.zscore(data[col].dropna())
        if np.max(np.abs(z_scores)) > 3:
            insights.append(f"- Kolom **{col}** memiliki nilai outlier berdasarkan Z-score")
            insight_found = True
        # Menampilkan informasi outlier berdasarkan IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            insights.append(f"- Kolom **{col}** memiliki nilai outlier berdasarkan IQR")
            insight_found = True

    # Tampilkan Insight atau Pesan Tidak Ada Insight
    if insight_found:
        for insight in insights:
            st.write(insight)
    else:
        st.info("Tidak ada insight yang ditemukan.")
else:
    st.info("Silakan unggah file untuk memulai analisis.")
