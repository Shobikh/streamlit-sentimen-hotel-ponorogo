import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import time
import json
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman Utama ---
st.set_page_config(
    page_title="Analisis Sentimen Hotel Ponorogo",
    page_icon="🏨",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT MODEL (DENGAN CACHING) ---
@st.cache_resource
def load_model_and_vectorizer(model_path, vectorizer_path):
    """Memuat model dan vectorizer dari file .pkl"""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None, None
    
@st.cache_data
def load_review_examples():
    with open('data/kata_berpengaruh.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# --- DATA TERPUSAT UNTUK KEDUA HOTEL ---
# Informasi spesifik, path gambar, dan path model untuk setiap hotel
data_hotel = {
    "Hotel Amaris Ponorogo": {
        "gambar": r'assets/amaris.png',
        "deskripsi": "Hotel Amaris Ponorogo merupakan bagian dari jaringan hotel Santika Indonesia yang dikenal dengan konsepnya yang modern, minimalis, dan cerdas. Berdasarkan ulasan dari berbagai platform, hotel ini unggul karena lokasinya yang sangat strategis di pusat kota, berdekatan dengan Ponorogo City Center (PCC) dan Alun-Alun. Pengunjung sering memuji kebersihan kamar dan standar pelayanan yang konsisten. Hotel ini menjadi pilihan utama bagi wisatawan bisnis dan keluarga yang mencari akomodasi modern dan efisien di jantung kota Ponorogo.",
        "wordcloud_positif": r"assets/wc_amaris_positif.png",
        "wordcloud_negatif": r"assets/wc_amaris_negatif.png",
        "model_path": r"models/model_sentimen_amaris.pkl",
        "vectorizer_path": r'models/tfidf_vectorizer_amaris.pkl',
        "evaluasi": {"Accuracy": 96, "Precision": 89, "Recall": 97, "F1-Score": 93},
        "distribusi": pd.DataFrame({'Sentimen': ['Positif', 'Negatif', 'Netral'], 'Jumlah Ulasan': [1646, 193, 153]}),
        "conf_matrix": [
            [152, 1, 0],
            [1, 192, 5],
            [33, 35, 1578]
        ],
        "classification_report": pd.DataFrame(
            {
                "precision": ["0.82", "0.84", "1.00"],
                "recall": ["0.99", "0.97", "0.96"],
                "f1-score": ["0.90", "0.90", "0.98"],
                "support": ["153 Sampel", "198 Sampel", "1646 Sampel"]
            },
            index=["Negatif", "Netral", "Positif"]
        )
    },
    "Hotel Maesa Ponorogo": {
        "gambar": r"assets/maesa.jpg",
        "deskripsi": "Hotel Maesa Ponorogo menawarkan pengalaman menginap yang berbeda dengan nuansa klasik dan sentuhan budaya Jawa yang kental. Dari rangkuman ulasan, keunggulan utama hotel ini adalah suasananya yang asri, kamar yang relatif lebih luas, dan keberadaan fasilitas kolam renang yang menjadi nilai tambah. Pengunjung sering menyebutkan keramahan staf yang terasa personal dan familier. Hotel ini menjadi pilihan menarik bagi mereka yang mencari ketenangan dan pengalaman menginap dengan kearifan lokal.",
        "wordcloud_positif": r"assets/wc_maesa_positif.png",
        "wordcloud_negatif": r"assets/wc_maesa_negatif.png",
        "model_path": r"models/model_sentimen_maesa.pkl",
        "vectorizer_path": r"models/tfidf_vectorizer_maesa.pkl",
        "evaluasi": {"Accuracy": 95, "Precision": 86, "Recall": 98, "F1-Score": 91},
        "distribusi": pd.DataFrame({'Sentimen': ['Positif', 'Negatif', 'Netral'], 'Jumlah Ulasan': [879, 104, 73]}),
        "conf_matrix": [
            [103, 1, 0],
            [0, 73, 0],
            [22, 22, 835]
        ],
        "classification_report": pd.DataFrame(
            {
                "precision": ["0.82", "0.76", "1.00"],
                "recall": ["0.99", "1.00", "0.95"],
                "f1-score": ["0.90", "0.86", "0.97"],
                "support": ["104 Sampel", "73 Sampel", "879 Sampel"],
            },
            index=["Negatif", "Netral", "Positif"]
        )
    }
}

contoh_ulasan_data = load_review_examples()

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.title("🏨")
    st.header("Dashboard Analisis Sentimen")
    
    pilihan_hotel = st.selectbox(
        "Pilih Hotel:",
        options=list(data_hotel.keys())
    )
    
    st.write("---")
    
    page = st.radio(
        "Pilih Menu:",
        ["📖 Latar Belakang & Profil", "⚙️ Cara Kerja Model", "📊 Hasil Analisis", "🧪 Coba Model"],
    )

hotel_terpilih = data_hotel[pilihan_hotel]


# --- KONTEN HALAMAN ---

if page == "📖 Latar Belakang & Profil":
    st.title(f"📖 Latar Belakang & Profil: {pilihan_hotel}")
    st.markdown("---")
    
    col1, col2 = st.columns([1.5, 2.5])
    
    with col1:
        try:
            st.image(hotel_terpilih["gambar"], caption=f"Tampak Depan {pilihan_hotel}", use_container_width=True, output_format='JPEG')
        except FileNotFoundError:
            st.warning(f"File gambar '{hotel_terpilih['gambar']}' tidak ditemukan.")
    
    with col2:
        st.subheader("Ringkasan Ulasan dari Berbagai Platform")
        deskripsi_hotel = hotel_terpilih["deskripsi"]
        st.markdown(f'<p style="text-align: justify;">{deskripsi_hotel}</p>', unsafe_allow_html=True)
        st.subheader("Detail Penelitian")
        st.success(
            f"""
            - **Sumber Data:** Traveloka, Agoda, Tiket.com, dan Google Maps.
            - **Waktu Pengambilan Data:** Snapshot per tanggal **31 Juli 2025**.
            - **Objek:** {pilihan_hotel}.
            """
        )

elif page == "⚙️ Cara Kerja Model":
    st.title("⚙️ Alur Kerja Model: Dari Ulasan Menjadi Sentimen")
    st.markdown("---")
    
    st.write(
        "Berikut adalah tahapan-tahapan yang dilalui secara sistematis untuk menganalisis sentimen dari setiap ulasan. "
        "Setiap tahap memiliki peran penting dalam menghasilkan prediksi yang akurat."
    )
    
    # --- TAHAP 1: PENGUMPULAN DATA ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center;'>📥</h1>", unsafe_allow_html=True)
        with col2:
            st.subheader("Tahap 1: Pengumpulan Data (Data Collection)")
            st.write(
                "Data ulasan mentah untuk **Hotel Amaris** dan **Hotel Maesa** dikumpulkan dari empat platform populer: "
                "**Traveloka, Agoda, Tiket.com, dan Google Maps**."
            )

    st.markdown("<h3 style='text-align: center;'>⬇️</h3>", unsafe_allow_html=True)

    # --- TAHAP 2: PREPROCESSING TEKS ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center;'>🧹</h1>", unsafe_allow_html=True)
        with col2:
            st.subheader("Tahap 2: Pembersihan Teks (Preprocessing)")
            st.write(
                "Teks ulasan yang masih berantakan dibersihkan agar seragam dan mudah diproses oleh model. "
                "Proses ini adalah fondasi dari analisis yang baik."
            )
            with st.expander("Lihat detail langkah-langkah preprocessing"):
                st.markdown("""
                - **Case Folding:** Semua huruf diubah menjadi huruf kecil (`lowercase`).
                - **Tokenizing:** Kalimat dipecah menjadi kata-kata individual (token).
                - **Stopword Removal:** Kata-kata umum yang tidak memiliki makna sentimen (seperti 'dan', 'di', 'yang') dihapus. Kata negasi seperti 'tidak' dan 'kurang' **tidak dihapus** untuk menjaga konteks.
                - **Stemming:** Setiap kata diubah ke bentuk dasarnya menggunakan algoritma stemmer untuk bahasa Indonesia (misal: 'pelayanannya' → 'layan').
                """)

    st.markdown("<h3 style='text-align: center;'>⬇️</h3>", unsafe_allow_html=True)
    
    # --- TAHAP 3: EKSTRAKSI FITUR ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center;'>🔢</h1>", unsafe_allow_html=True)
        with col2:
            st.subheader("Tahap 3: Ekstraksi Fitur (TF-IDF)")
            st.write(
                "Kata-kata yang sudah bersih diubah menjadi angka yang dapat 'dipahami' oleh komputer. "
                "Metode **TF-IDF (Term Frequency-Inverse Document Frequency)** digunakan untuk mengukur seberapa penting sebuah kata dalam sebuah ulasan."
            )
            st.code("vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)", language="python")

    st.markdown("<h3 style='text-align: center;'>⬇️</h3>", unsafe_allow_html=True)

    # --- TAHAP 4: PELATIHAN MODEL ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
        with col2:
            st.subheader("Tahap 4: Pelatihan Model (Training)")
            st.write(
                "Data numerik dari TF-IDF digunakan untuk melatih model **Naive Bayes**. "
                "Model 'belajar' mengenali pola kata yang cenderung muncul di ulasan positif dan negatif."
            )
            st.code("model = MultinomialNB()\nmodel.fit(X_train, y_train)", language="python")

    st.markdown("<h3 style='text-align: center;'>⬇️</h3>", unsafe_allow_html=True)

    # --- TAHAP 5: EVALUASI & PREDIKSI ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center;'>✅</h1>", unsafe_allow_html=True)
        with col2:
            st.subheader("Tahap 5: Evaluasi dan Penggunaan Model")
            st.write(
                "Model yang sudah terlatih kemudian dievaluasi kinerjanya menggunakan metrik seperti Akurasi dan Confusion Matrix. "
                "Model inilah yang akhirnya digunakan di halaman **'Coba Model'** untuk memprediksi sentimen ulasan baru."
            )

elif page == "📊 Hasil Analisis":
    st.title(f"📊 Hasil Analisis Mendalam: {pilihan_hotel}")
    st.markdown("---")
    
    st.subheader("Performa Model (Berdasarkan Data Aktual)")
    eval_data = hotel_terpilih["evaluasi"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Accuracy", value=f"{eval_data['Accuracy']}%", delta_color="off")
    col2.metric(label="Precision (Macro Avg)", value=f"{eval_data['Precision']}%", delta_color="off")
    col3.metric(label="Recall (Macro Avg)", value=f"{eval_data['Recall']}%", delta_color="off")
    col4.metric(label="F1-Score (Macro Avg)", value=f"{eval_data['F1-Score']}%", delta_color="off")
    
    st.markdown("---")
    st.subheader("💡 Kata Kunci Paling Berpengaruh & Contoh Ulasannya")
    st.info(
        "**Penjelasan:** Klik pada setiap kata untuk melihat contoh nyata dari ulasan pengguna yang mengandung kata tersebut. "
        "Ini adalah kata-kata yang paling kuat memengaruhi model dalam menentukan sentimen.",
        icon="ℹ️"
    )
    
    ulasan_hotel_terpilih = contoh_ulasan_data.get(pilihan_hotel, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h5 style='text-align: center; color: green;'>👍 Sentimen Positif</h5>", unsafe_allow_html=True)
        st.write("") # Memberi spasi
        positif_examples = ulasan_hotel_terpilih.get("Positif", {})

        for word, review_data in positif_examples.items():
            with st.expander(f"**{word.capitalize()}**"):
                st.write(f"_{review_data['review']}_")

                #Keterangan tambahan
                reviewer_name = review_data.get("reviewer","Anonymous")
                platform = review_data.get("platform")
                rating = review_data.get("rating")
                st.caption(f"**Diulas oleh {reviewer_name} di {platform} dengan rating {rating}**")
            

    with col2:
        st.markdown("<h5 style='text-align: center; color: red;'>👎 Sentimen Negatif</h5>", unsafe_allow_html=True)
        st.write("") # Memberi spasi
        negatif_examples = ulasan_hotel_terpilih.get("Negatif", {})

        for word, review_data in negatif_examples.items():
            with st.expander(f"**{word.capitalize()}**"):
                st.write(f"_{review_data['review']}_")

                #Keterangan tambahan
                reviewer_name = review_data.get("reviewer","Anonymous")
                platform = review_data.get("platform")
                rating = review_data.get("rating")
                st.caption(f"**Diulas oleh {reviewer_name} di {platform} dengan rating {rating}**")

    st.markdown("---")
    
    with st.expander("Lihat Detail Performa dengan Confusion Matrix dan Classification Report"):
        st.subheader("Visualisasi Confusion Matrix")
        st.write(
            "Confusion Matrix menunjukkan seberapa baik model dapat membedakan antar kelas. "
            "Sumbu vertikal (kiri) adalah **Label Asli**, dan sumbu horizontal (atas) adalah **Label Prediksi**. "
            "Angka pada diagonal utama (biru tua) menunjukkan prediksi yang benar."
        )
    
        # Membuat DataFrame dari data
        cm_data = hotel_terpilih["conf_matrix"]
        labels = ['Negatif', 'Netral', 'Positif']
        
        df_cm = pd.DataFrame(cm_data, index=labels, columns=labels)
        
        # Menampilkan DataFrame dengan gaya heatmap
        st.dataframe(df_cm.style.background_gradient(cmap='Blues', axis=None))
        st.subheader("Classification Report")
        st.write(
                    "Classification Report menunjukkan performa model dalam mengklasifikasikan label "
                    "berdasarkan tingkat presisi (precision), sensitivitas (recall), keseimbangan keduanya (f1-score), dan jumlah sampel (support)"
                )
        st.table(hotel_terpilih["classification_report"])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Sentimen (Data Asli)")
        st.bar_chart(hotel_terpilih["distribusi"].set_index('Sentimen'))
        
    with col2:
        st.subheader("Kata Kunci Teratas (WordCloud)")
        tab1, tab2 = st.tabs(["👍 Positif", "👎 Negatif"])
        with tab1:
            try:
                st.image(hotel_terpilih["wordcloud_positif"], use_container_width=True)
            except FileNotFoundError:
                st.warning(f"File gambar '{hotel_terpilih['wordcloud_positif']}' tidak ditemukan.")
        with tab2:
            try:
                st.image(hotel_terpilih["wordcloud_negatif"], use_container_width=True)
            except FileNotFoundError:
                st.warning(f"File gambar '{hotel_terpilih['wordcloud_negatif']}' tidak ditemukan.")

elif page == "🧪 Coba Model":
    # Kode untuk halaman ini tidak berubah
    st.title(f"🧪 Uji Coba Model Klasifikasi: {pilihan_hotel}")
    st.markdown("---")
    st.info(f"Anda sedang menguji model Naive Bayes* yang dilatih khusus untuk **{pilihan_hotel}**.")
    st.caption("*Cara Kerja Model Naive Bayes Didasarkan pada Probabilitas (Teorema Bayes)")

    model, vectorizer = load_model_and_vectorizer(hotel_terpilih['model_path'], hotel_terpilih['vectorizer_path'])

    if model is None or vectorizer is None:
        st.error(f"Gagal memuat file model. Pastikan file '{hotel_terpilih['model_path']}' dan '{hotel_terpilih['vectorizer_path']}' ada di folder aplikasi.")
    else:
        user_input = st.text_area("Ketik ulasan Anda di sini:", "Kamarnya sangat nyaman dan bersih, tapi sarapannya kurang bervariasi.", height=150)

        if st.button("✨ Analisis Sekarang!", type="primary", use_container_width=True):
            if user_input:
                with st.spinner('Menganalisis sentimen...'):
                    time.sleep(1)
                    vectorized_input = vectorizer.transform([user_input])
                    prediction = model.predict(vectorized_input)
                    sentiment = prediction[0]

                    st.markdown("---")
                    st.subheader("Hasil Prediksi:")
                    
                    if str(sentiment).lower() == "positif":
                        st.success(f"**Sentimen: Positif** 👍")
                        st.balloons()
                    elif str(sentiment).lower() == "negatif":
                        st.error(f"**Sentimen: Negatif** 👎")
                    else:
                        st.info(f"**Sentimen: Netral** 😐")
            else:
                st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
