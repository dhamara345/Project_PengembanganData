import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import io

# Sklearn Core, Preprocessing & Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- GABUNGAN MODEL LENGKAP DARI CODINGAN DOSEN ----
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# Boosting Lanjutan (Eksternal Ekosistem)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# CONFIG & LOAD DATASET
# ==========================================
st.set_page_config(page_title="Automated ML Benchmarking", layout="wide")
st.title("💳 Automated ML Benchmarking Dashboard (Full Dosen Style)")
st.write("Eksperimen komparasi model skala besar menggunakan arsitektur Pipeline, penanganan error otomatis, dan metrik ROC-AUC.")

# Folder untuk menyimpan model pickle
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_data
def load_data():
    return pd.read_csv("UCI_Credit_Card.csv")

try:
    df = load_data()

    # ---- SIDEBAR: NAVIGASI UTAMA ----
    st.sidebar.header("🧭 Navigasi Dashboard")
    menu = st.sidebar.radio(
        "Pilih Halaman Proyek:",
        [
            "1. Eksplorasi Data",
            "2. Definisi Fitur & Pipeline",
            "3. Benchmarking Akhir (Fungsi Dosen)",
            "4. Simpan & Load Model (Pickle)"
        ]
    )

    # Kunci Definisi Kolom secara Global agar Sinkron di Semua Menu
    kolom_numerik_kontinu = [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    kolom_kategorikal_status = [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]

    # ==========================================
    # --- MENU 1: EKSPLORASI DATA ---
    # ==========================================
    if menu == "1. Eksplorasi Data":
        st.subheader("📊 Analisis Karakteristik & Fitur Dataset")

        col_box1, col_box2, col_box3 = st.columns(3)
        with col_box1:
            st.metric("Total Baris Data", f"{df.shape[0]}")
        with col_box2:
            st.metric("Total Kolom (Fitur)", f"{df.shape[1]}")
        with col_box3:
            distribusi = df['default.payment.next.month'].value_counts()
            st.metric("Rasio Gagal Bayar (Class 1)", f"{(distribusi[1]/df.shape[0])*100:.1f}%")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "📋 Daftar Kolom & Sampel",
            "📈 Statistik Deskriptif",
            "👥 1. Demografi vs Risiko Default",
            "⚡ 2. Analisis Prediktor Terkuat"
        ])

        with sub_tab1:
            st.markdown("**Preview 5 Data Teratas:**")
            st.dataframe(df.head(5), use_container_width=True)
            st.markdown("### 📑 Keterangan Variabel Dataset (Metadata):")
            st.markdown("""
            * **ID**: Identifikasi unik setiap nasabah.
            * **LIMIT_BAL**: Jumlah kredit yang diberikan (Dolar NT), termasuk kredit individu/keluarga.
            * **SEX**: Jenis Kelamin (1 = Laki-laki, 2 = Perempuan).
            * **EDUCATION**: Tingkat Pendidikan (1 = S2/S3, 2 = S1, 3 = SMA, 4 = Lainnya, 5/6 = Tidak Diketahui).
            * **MARRIAGE**: Status Pernikahan (1 = Menikah, 2 = Lajang, 3 = Lainnya).
            * **AGE**: Usia nasabah (Tahun).
            * **PAY_0 s/d PAY_6**: Status pembayaran bulanan (September - April 2005).
            * **BILL_AMT1 s/d BILL_AMT6**: Jumlah tagihan kartu kredit bulanan (September - April 2005).
            * **PAY_AMT1 s/d PAY_AMT6**: Jumlah pembayaran nominal bulanan sebelumnya (September - April 2005).
            * **default.payment.next.month (Target)**: Status gagal bayar bulan depan (1 = Ya/Gagal Bayar, 0 = Tidak/Lancar).
            """)

        with sub_tab2:
            st.markdown("**Statistik Deskriptif Dataset:**")
            st.dataframe(df.describe(), use_container_width=True)

        with sub_tab3:
            st.markdown("### 👥 Probabilitas Gagal Bayar Berdasarkan Karakteristik Demografi")
            st.info("💡 **Ide Eksplorasi:** *How does the probability of default payment vary by categories of different demographic variables?*")

            demo_var = st.selectbox("Pilih Variabel Demografi untuk Ditampilkan:", ["SEX", "EDUCATION", "MARRIAGE"])
            mapping_labels = {
                "SEX": {1: "1. Male", 2: "2. Female"},
                "EDUCATION": {1: "1. Graduate School", 2: "2. University", 3: "3. High School", 4: "4. Others", 5: "5. Unknown", 6: "6. Unknown"},
                "MARRIAGE": {1: "1. Married", 2: "2. Single", 3: "3. Others"}
            }

            df_demo = df.copy()
            df_demo[demo_var] = df_demo[demo_var].map(mapping_labels[demo_var])

            df_grouped = df_demo.groupby(demo_var)['default.payment.next.month'].mean().reset_index()
            df_grouped['default.payment.next.month'] *= 100
            df_grouped = df_grouped.sort_values(by='default.payment.next.month', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(x=demo_var, y='default.payment.next.month', data=df_grouped, palette='viridis', ax=ax)
            ax.set_ylabel("Rasio Gagal Bayar (%)")
            ax.set_xlabel(f"Kategori {demo_var}")
            ax.set_title(f"Persentase Risiko Gagal Bayar Berdasarkan {demo_var}")
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        with sub_tab4:
            st.markdown("### ⚡ Identifikasi Fitur Prediktor Terkuat")
            st.info("💡 **Ide Eksplorasi:** *Which variables are the strongest predictors of default payment?*")

            matrix_korelasi = df.corr()['default.payment.next.month'].drop(['ID', 'default.payment.next.month']).reset_index()
            matrix_korelasi.columns = ['Nama Fitur', 'Nilai Korelasi']
            matrix_korelasi['Korelasi Absolut'] = matrix_korelasi['Nilai Korelasi'].abs()
            matrix_korelasi = matrix_korelasi.sort_values(by='Korelasi Absolut', ascending=False).reset_index(drop=True)

            col_graph, col_table = st.columns([3, 2])
            with col_table:
                st.markdown("**Tabel Urutan Fitur Paling Berpengaruh:**")
                st.dataframe(matrix_korelasi[['Nama Fitur', 'Nilai Korelasi']], use_container_width=True)
            with col_graph:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Nilai Korelasi', y='Nama Fitur', data=matrix_korelasi.head(10), palette='coolwarm', ax=ax)
                ax.set_title("Top 10 Fitur Prediktor Terkuat (Berdasarkan Korelasi)")
                ax.set_xlabel("Koefisien Korelasi")
                ax.set_ylabel("Nama Fitur")
                st.pyplot(fig)
                plt.close(fig)

    # ==========================================
    # --- MENU 2: DEFINISI FITUR & PIPELINE ---
    # ==========================================
    elif menu == "2. Definisi Fitur & Pipeline":
        st.subheader("⚙️ Pemetaan Arsitektur Pra-pemrosesan Data (Pipeline)")
        st.write("Halaman ini mendefinisikan pembagian perlakuan data input (Fitur) dan sasaran prediksi (Target).")

        col_def1, col_def2, col_def3 = st.columns(3)
        with col_def1:
            st.info("### 📈 1. Fitur Numerik Kontinu")
            st.write("Skala nilai besar di-standardisasi lewat fungsi `StandardScaler()`.")
            st.json(kolom_numerik_kontinu)
        with col_def2:
            st.warning("### 🔢 2. Fitur Kategorikal & Status")
            st.write("Fitur kode identifikasi dilewatkan utuh lewat perintah `passthrough`.")
            st.json(kolom_kategorikal_status)
        with col_def3:
            st.success("### 🎯 3. Target Variable (Label Y)")
            st.code("default.payment.next.month", language="text")
            st.markdown("""
            **Interpretasi Nilai Target:**
            * **0**: Nasabah Lancar (Tidak Default)
            * **1**: Nasabah Gagal Bayar (Default)
            """)

        st.write("---")
        st.markdown("### 🛠️ Alur Kerja Gabungan dalam Pipeline")
        st.code("""
# 1. Pisahkan Fitur (X) dan Target (y) dari Dataset
X = df.drop(columns=['ID', 'default.payment.next.month'])
y = df['default.payment.next.month']

# 2. Arsitektur ColumnTransformer untuk Fitur X
preprocessor = ColumnTransformer(
    transformers=[
        ('num_scale', StandardScaler(), kolom_numerik_kontinu),
        ('cat_keep', 'passthrough', kolom_kategorikal_status)
    ]
)

# 3. Penggabungan Otomatis ke Model Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_terpilih)
])
        """, language="python")

    # ==========================================
    # --- MENU 3: BENCHMARKING AKHIR ---
    # ==========================================
    elif menu == "3. Benchmarking Akhir (Fungsi Dosen)":
        st.sidebar.write("---")
        st.sidebar.header("🎛️ Konfigurasi Model & Partisi")
        test_size = st.sidebar.slider("Ukuran Data Test (%)", 10, 50, 20, step=5) / 100
        random_state_val = st.sidebar.number_input("Random State Seed", value=10)

        all_models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', solver='liblinear', random_state=int(random_state_val)),
            "SGD Classifier": SGDClassifier(class_weight='balanced', random_state=int(random_state_val), loss='log_loss'),
            "Ridge Classifier": RidgeClassifier(class_weight='balanced', random_state=int(random_state_val)),
            "Passive Aggressive": PassiveAggressiveClassifier(class_weight='balanced', random_state=int(random_state_val), max_iter=1000),
            "Gaussian NB": GaussianNB(),
            "Multinomial NB": MultinomialNB(),
            "Bernoulli NB": BernoulliNB(),
            "Complement NB": ComplementNB(),
            "SVC": SVC(probability=True, class_weight='balanced', random_state=int(random_state_val)),
            "Linear SVC": LinearSVC(class_weight='balanced', random_state=int(random_state_val), dual=False),
            "NuSVC": NuSVC(probability=True, class_weight='balanced', random_state=int(random_state_val)),
            "Decision Tree": DecisionTreeClassifier(random_state=int(random_state_val)),
            "Extra Tree (single)": ExtraTreeClassifier(random_state=int(random_state_val)),
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=int(random_state_val)),
            "Gradient Boosting": GradientBoostingClassifier(random_state=int(random_state_val)),
            "AdaBoost": AdaBoostClassifier(random_state=int(random_state_val)),
            "Bagging": BaggingClassifier(random_state=int(random_state_val)),
            "Extra Trees": ExtraTreesClassifier(random_state=int(random_state_val)),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=int(random_state_val)),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Nearest Centroid": NearestCentroid(),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "MLP Neural Network": MLPClassifier(random_state=int(random_state_val), max_iter=500),
            "XGBoost": XGBClassifier(random_state=int(random_state_val), eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=int(random_state_val), is_unbalance=True, verbose=-1),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=int(random_state_val), auto_class_weights='Balanced'),
            "Gaussian Process (⚠️ Sangat Lambat)": GaussianProcessClassifier(random_state=int(random_state_val))
        }

        default_pilihan = [m for m in all_models.keys() if "Sangat Lambat" not in m and "NuSVC" not in m and "SVC" not in m]
        model_terpilih = st.sidebar.multiselect(
            "Pilih Model untuk Dieksekusi:",
            list(all_models.keys()),
            default=default_pilihan
        )

        # Opsi simpan model setelah benchmarking
        st.sidebar.write("---")
        st.sidebar.header("💾 Opsi Pickle")
        simpan_semua = st.sidebar.checkbox("Simpan semua model ke .pkl", value=False)
        simpan_terbaik = st.sidebar.checkbox("Simpan hanya model terbaik (ROC-AUC tertinggi)", value=True)

        st.subheader("🚀 Papan Eksekusi Benchmarking")

        X = df.drop(columns=['ID', 'default.payment.next.month'])
        y = df['default.payment.next.month']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state_val), stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_scale', StandardScaler(), kolom_numerik_kontinu),
                ('cat_keep', 'passthrough', kolom_kategorikal_status)
            ]
        )

        if st.button("Mulai Proses Benchmarking ⚡"):
            if not model_terpilih:
                st.error("Silakan pilih minimal 1 model di sidebar sebelum memulai!")
            else:
                results = []
                trained_pipelines = {}  # Menyimpan pipeline yang sudah dilatih
                log_container = st.expander("📄 Log Proses Eksekusi (Real-time)", expanded=True)
                progress_bar = st.progress(0)
                total_pilihan = len(model_terpilih)

                for idx, name in enumerate(model_terpilih):
                    model_obj = all_models[name]
                    log_container.write(f"🔄 **Memproses:** {name}...")

                    try:
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', model_obj)
                        ])

                        start_t = time.time()
                        pipeline.fit(X_train, y_train)
                        durasi = time.time() - start_t

                        y_pred = pipeline.predict(X_test)

                        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                            y_scores = pipeline.predict_proba(X_test)[:, 1]
                        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
                            y_scores = pipeline.decision_function(X_test)
                        else:
                            y_scores = None

                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        roc_auc = roc_auc_score(y_test, y_scores) if y_scores is not None else np.nan

                        results.append({
                            'Model': name,
                            'Accuracy': acc,
                            'Precision': prec,
                            'Recall': rec,
                            'F1-Score': f1,
                            'ROC-AUC': roc_auc,
                            'Duration': f"{durasi:.2f} s"
                        })
                        trained_pipelines[name] = pipeline  # Simpan pipeline terlatih

                        # Simpan semua model jika opsi diaktifkan
                        if simpan_semua:
                            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("⚠️", "").strip()
                            pkl_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
                            with open(pkl_path, 'wb') as f:
                                pickle.dump(pipeline, f)

                        log_container.success(f"✅ **Selesai:** {name} ({durasi:.2f} detik)")

                    except Exception as e:
                        alasan = str(e).split(':')[0]
                        log_container.error(f"❌ **Gagal:** {name} (Alasan: {alasan})")
                        results.append({
                            'Model': name,
                            'Accuracy': np.nan,
                            'Precision': np.nan,
                            'Recall': np.nan,
                            'F1-Score': np.nan,
                            'ROC-AUC': np.nan,
                            'Duration': "Gagal"
                        })

                    progress_bar.progress((idx + 1) / total_pilihan)

                results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
                results_df.insert(0, 'Rank', results_df.index + 1)

                st.write("---")
                st.subheader("📋 Leaderboard Hasil Berdasarkan Urutan ROC-AUC Dosen")

                df_display = results_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")

                st.dataframe(df_display, use_container_width=True)

                pemenang = results_df.iloc[0]['Model']
                skor_auc = results_df.iloc[0]['ROC-AUC']

                if not pd.isna(skor_auc):
                    st.success(f"🏆 Model Terbaik menurut standar dosen adalah **{pemenang}** dengan skor **ROC-AUC: {skor_auc*100:.2f}%**.")

                    # ---- SIMPAN MODEL TERBAIK ----
                    if simpan_terbaik and pemenang in trained_pipelines:
                        best_pipeline = trained_pipelines[pemenang]
                        safe_name = pemenang.replace(" ", "_").replace("(", "").replace(")", "").replace("⚠️", "").strip()
                        best_pkl_path = os.path.join(MODEL_DIR, f"BEST_{safe_name}.pkl")

                        with open(best_pkl_path, 'wb') as f:
                            pickle.dump(best_pipeline, f)

                        # Buat file untuk diunduh langsung dari browser
                        buffer = io.BytesIO()
                        pickle.dump(best_pipeline, buffer)
                        buffer.seek(0)

                        st.write("---")
                        st.subheader("💾 Ekspor Model Terbaik")
                        col_info, col_dl = st.columns([3, 1])
                        with col_info:
                            st.info(f"""
                            **Model tersimpan:** `{safe_name}.pkl`  
                            **Lokasi di server:** `{best_pkl_path}`  
                            **Model ini sudah include preprocessor Pipeline**, sehingga siap langsung digunakan untuk prediksi data baru tanpa perlu scaling ulang.
                            """)
                        with col_dl:
                            st.download_button(
                                label="⬇️ Download Model (.pkl)",
                                data=buffer,
                                file_name=f"BEST_{safe_name}.pkl",
                                mime="application/octet-stream"
                            )

                    if simpan_semua:
                        st.info(f"✅ Semua {len(trained_pipelines)} model berhasil disimpan ke folder `{MODEL_DIR}/`")

    # ==========================================
    # --- MENU 4: SIMPAN & LOAD MODEL (PICKLE) ---
    # ==========================================
    elif menu == "4. Simpan & Load Model (Pickle)":
        st.subheader("🗂️ Manajemen Model Pickle — Simpan & Load")

        tab_load_file, tab_load_lokal, tab_prediksi = st.tabs([
            "📤 Upload & Load Model (.pkl)",
            "📁 Model Tersimpan di Server",
            "🔮 Prediksi Data Baru"
        ])

        # ---- TAB 1: Upload file pkl dari komputer ----
        with tab_load_file:
            st.markdown("### Upload File Model (.pkl)")
            st.write("Upload file `.pkl` yang sebelumnya disimpan dari hasil benchmarking untuk digunakan kembali.")

            uploaded_pkl = st.file_uploader("Pilih file model (.pkl)", type=["pkl"])

            if uploaded_pkl is not None:
                try:
                    loaded_pipeline = pickle.load(uploaded_pkl)
                    st.session_state['loaded_model'] = loaded_pipeline
                    st.session_state['loaded_model_name'] = uploaded_pkl.name
                    st.success(f"✅ Model **`{uploaded_pkl.name}`** berhasil dimuat!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Informasi Model:**")
                        st.json({
                            "Nama File": uploaded_pkl.name,
                            "Tipe Objek": str(type(loaded_pipeline)),
                            "Steps Pipeline": str([s[0] for s in loaded_pipeline.steps]) if hasattr(loaded_pipeline, 'steps') else "Bukan Pipeline"
                        })
                    with col2:
                        st.markdown("**Classifier dalam Pipeline:**")
                        if hasattr(loaded_pipeline, 'named_steps'):
                            classifier = loaded_pipeline.named_steps.get('classifier', None)
                            st.code(str(classifier), language="text")

                except Exception as e:
                    st.error(f"❌ Gagal memuat model: {e}")

        # ---- TAB 2: Daftar model tersimpan di server ----
        with tab_load_lokal:
            st.markdown("### Model yang Tersimpan di Server")

            pkl_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')] if os.path.exists(MODEL_DIR) else []

            if not pkl_files:
                st.warning("⚠️ Belum ada model yang tersimpan. Jalankan Benchmarking terlebih dahulu dengan opsi simpan diaktifkan.")
            else:
                st.success(f"Ditemukan **{len(pkl_files)} model** tersimpan di folder `{MODEL_DIR}/`")

                for fname in pkl_files:
                    fpath = os.path.join(MODEL_DIR, fname)
                    fsize = os.path.getsize(fpath) / 1024  # KB

                    col_name, col_size, col_load, col_dl = st.columns([3, 1, 1, 1])
                    with col_name:
                        st.write(f"📦 `{fname}`")
                    with col_size:
                        st.write(f"{fsize:.1f} KB")
                    with col_load:
                        if st.button("Load", key=f"load_{fname}"):
                            with open(fpath, 'rb') as f:
                                loaded_pipeline = pickle.load(f)
                            st.session_state['loaded_model'] = loaded_pipeline
                            st.session_state['loaded_model_name'] = fname
                            st.success(f"Model `{fname}` aktif!")
                    with col_dl:
                        with open(fpath, 'rb') as f:
                            st.download_button(
                                label="⬇️ DL",
                                data=f,
                                file_name=fname,
                                mime="application/octet-stream",
                                key=f"dl_{fname}"
                            )

                if 'loaded_model_name' in st.session_state:
                    st.info(f"🟢 **Model aktif saat ini:** `{st.session_state['loaded_model_name']}`")

        # ---- TAB 3: Prediksi Data Baru ----
        with tab_prediksi:
            st.markdown("### 🔮 Prediksi Nasabah Baru")

            if 'loaded_model' not in st.session_state:
                st.warning("⚠️ Belum ada model yang di-load. Silakan load model di tab sebelumnya terlebih dahulu.")
            else:
                st.success(f"✅ Menggunakan model: **`{st.session_state.get('loaded_model_name', 'Unknown')}`**")
                st.write("---")

                st.markdown("#### Input Data Nasabah Baru")
                st.caption("Isi form di bawah ini untuk memprediksi apakah nasabah akan gagal bayar bulan depan.")

                # Form input data nasabah
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("**📋 Data Pribadi**")
                    limit_bal = st.number_input("LIMIT_BAL (Limit Kredit)", min_value=0, value=50000, step=5000)
                    age = st.number_input("AGE (Usia)", min_value=18, max_value=100, value=30)
                    sex = st.selectbox("SEX", options=[1, 2], format_func=lambda x: "1 - Laki-laki" if x == 1 else "2 - Perempuan")
                    education = st.selectbox("EDUCATION", options=[1, 2, 3, 4], format_func=lambda x: {1: "1-Graduate School", 2: "2-University", 3: "3-High School", 4: "4-Others"}[x])
                    marriage = st.selectbox("MARRIAGE", options=[1, 2, 3], format_func=lambda x: {1: "1-Married", 2: "2-Single", 3: "3-Others"}[x])

                with col_b:
                    st.markdown("**💳 Riwayat Pembayaran (PAY)**")
                    st.caption("(-1=tepat waktu, 1=telat 1bln, 2=telat 2bln, dst)")
                    pay_0 = st.slider("PAY_0 (Sep)", -2, 8, 0)
                    pay_2 = st.slider("PAY_2 (Agu)", -2, 8, 0)
                    pay_3 = st.slider("PAY_3 (Jul)", -2, 8, 0)
                    pay_4 = st.slider("PAY_4 (Jun)", -2, 8, 0)
                    pay_5 = st.slider("PAY_5 (Mei)", -2, 8, 0)
                    pay_6 = st.slider("PAY_6 (Apr)", -2, 8, 0)

                with col_c:
                    st.markdown("**🧾 Tagihan & Pembayaran**")
                    bill_amt1 = st.number_input("BILL_AMT1 (Sep)", value=10000, step=1000)
                    bill_amt2 = st.number_input("BILL_AMT2 (Agu)", value=9000, step=1000)
                    bill_amt3 = st.number_input("BILL_AMT3 (Jul)", value=8500, step=1000)
                    bill_amt4 = st.number_input("BILL_AMT4 (Jun)", value=8000, step=1000)
                    bill_amt5 = st.number_input("BILL_AMT5 (Mei)", value=7500, step=1000)
                    bill_amt6 = st.number_input("BILL_AMT6 (Apr)", value=7000, step=1000)

                col_d, col_e = st.columns(2)
                with col_d:
                    pay_amt1 = st.number_input("PAY_AMT1 (Sep)", value=2000, step=500)
                    pay_amt2 = st.number_input("PAY_AMT2 (Agu)", value=2000, step=500)
                    pay_amt3 = st.number_input("PAY_AMT3 (Jul)", value=2000, step=500)
                with col_e:
                    pay_amt4 = st.number_input("PAY_AMT4 (Jun)", value=2000, step=500)
                    pay_amt5 = st.number_input("PAY_AMT5 (Mei)", value=2000, step=500)
                    pay_amt6 = st.number_input("PAY_AMT6 (Apr)", value=2000, step=500)

                st.write("---")
                if st.button("🔮 Prediksi Sekarang", type="primary"):
                    # Susun input sesuai urutan kolom X
                    input_data = pd.DataFrame([{
                        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education,
                        'MARRIAGE': marriage, 'AGE': age,
                        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3,
                        'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
                        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
                        'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
                        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
                        'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6,
                    }])

                    try:
                        model = st.session_state['loaded_model']
                        prediksi = model.predict(input_data)[0]

                        prob = None
                        if hasattr(model.named_steps['classifier'], "predict_proba"):
                            prob = model.predict_proba(input_data)[0][1]

                        st.write("---")
                        if prediksi == 1:
                            st.error(f"⚠️ **PREDIKSI: GAGAL BAYAR (DEFAULT)**")
                            if prob is not None:
                                st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")
                            st.markdown("""
                            > Nasabah ini diprediksi **berisiko tinggi** untuk gagal bayar pada bulan depan. 
                            > Disarankan dilakukan evaluasi kredit lebih lanjut.
                            """)
                        else:
                            st.success(f"✅ **PREDIKSI: LANCAR (TIDAK DEFAULT)**")
                            if prob is not None:
                                st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")
                            st.markdown("""
                            > Nasabah ini diprediksi **tidak berisiko** gagal bayar pada bulan depan.
                            """)

                        with st.expander("📊 Lihat Data Input yang Digunakan"):
                            st.dataframe(input_data, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Prediksi gagal: {e}")

except FileNotFoundError:
    st.error("❌ File `UCI_Credit_Card.csv` tidak ditemukan di folder proyek!")
