import streamlit as st
import joblib
import pandas as pd
import altair as alt

# Konfigurasi dasar halaman
st.set_page_config(
    page_title="Prediksi Pembelian Produk",
    page_icon="🛍️",
    layout="wide",
)

# Muat model dan parameter scaling
gnb = joblib.load("naive_bayes_model.pkl")
scaling_params = joblib.load("scaling_params.pkl")

# Gaya kustom untuk tampilan lebih modern
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 10% 20%, #e0f2fe 0%, #f8fafc 25%, #eef2ff 55%, #e0f7fa 100%);
    }
    .block-container {
        padding: 1.5rem 2.25rem 3rem;
        max-width: 1200px;
    }
    .hero {
        background: linear-gradient(120deg, #0f172a, #1e293b);
        color: #f8fafc;
        padding: 1.75rem 1.9rem;
        border-radius: 18px;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.1);
        border: 1px solid #e2e8f0;
        backdrop-filter: blur(4px);
    }
    .result-pill {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
    }
    .pills-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 0;
    }
    .pill {
        background: rgba(255,255,255,0.12);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.18);
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(160deg, #f8fafc 0%, #ecfeff 60%, #e0f2fe 100%);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem;
    }
    hr {
        border: none;
        border-bottom: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom: 0.2rem;">Prediksi Pembelian Produk</h1>
        <p style="margin: 0; opacity: 0.8;">
            Masukkan profil singkat pengguna untuk memprediksi kemungkinan pembelian.
        </p>
        <div class="pills-row">
            <span class="pill">⚡ Gaussian Naive Bayes</span>
            <span class="pill">📊 3 fitur utama</span>
            <span class="pill">🚀 Realtime inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Input Data Pengguna")

# Form input untuk data pengguna
gender = st.sidebar.selectbox("Pilih Gender", ["Female", "Male"])
age = st.sidebar.slider("Usia Pengguna", 18, 100, 30)
salary = st.sidebar.slider("Gaji Pengguna (per tahun)", 0, 200000, 50000, step=1000)

# Encoding Gender
gender_encoded = 1 if gender == "Male" else 0

# Scaling sesuai dengan data training
age_scaled = (age - scaling_params["mean_age"]) / scaling_params["std_age"]
salary_scaled = (salary - scaling_params["mean_salary"]) / scaling_params["std_salary"]

# DataFrame input
input_data = pd.DataFrame(
    {
        "Gender": [gender_encoded],
        "Age": [age_scaled],
        "EstimatedSalary": [salary_scaled],
    }
)

# Prediksi kelas dan probabilitas
probabilities = gnb.predict_proba(input_data)
threshold = 0.5
prediction = 1 if probabilities[0][1] >= threshold else 0

prob_purchase = probabilities[0][1] * 100
prob_no_purchase = probabilities[0][0] * 100

# Layout dua kolom: hasil & rincian
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.markdown(
            '<span class="result-pill" style="background:#dcfce7;color:#14532d;">'
            "Pembelian Produk (Ya)"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="result-pill" style="background:#fee2e2;color:#7f1d1d;">'
            "Tidak Membeli Produk"
            "</span>",
            unsafe_allow_html=True,
        )

    st.write(" ")
    st.write("Probabilitas:")
    st.progress(prob_purchase / 100)
    st.write(f"Peluang Pembelian: **{prob_purchase:.2f}%**")
    st.write(f"Peluang Tidak Pembelian: **{prob_no_purchase:.2f}%**")

    # Visualisasi probabilitas yang lebih menarik
    chart_data = pd.DataFrame(
        {
            "Kategori": ["Pembelian", "Tidak Membeli"],
            "Probabilitas (%)": [prob_purchase, prob_no_purchase],
            "Warna": ["#22c55e", "#ef4444"],
        }
    )

    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Probabilitas (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Kategori", sort="-x"),
            color=alt.Color("Warna", scale=None),
            tooltip=["Kategori", "Probabilitas (%)"],
        )
        .properties(height=140)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Input")
    st.metric("Gender", gender)
    st.metric("Usia", f"{age} tahun")
    st.metric("Gaji (per tahun)", f"${salary:,.0f}")
    st.caption(
        "Nilai sudah disesuaikan dengan skala data pelatihan agar model lebih akurat."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Tabs informasi tambahan
about_tab, tips_tab = st.tabs(["ℹ️ Tentang Model", "💡 Tips Menggunakan"])

with about_tab:
    st.markdown(
        """
        Model menggunakan Gaussian Naive Bayes yang ringan sehingga inference berlangsung instan.
        Fitur yang dipakai:
        - Gender (di-encode biner)
        - Usia (distandarisasi)
        - Gaji (distandarisasi)
        """
    )

with tips_tab:
    st.markdown(
        """
        - Gunakan slider untuk melihat bagaimana usia dan gaji memengaruhi probabilitas.
        - Input gender memengaruhi prior, namun pengaruh utama ada pada gaji dan usia.
        - Simpan angka gaji dalam rentang realistis agar interpretasi tetap akurat.
        """
    )
