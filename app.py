import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0D1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"] label {
        color: #8B949E !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E6EDF3 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3 !important;
    }
    h1 { font-weight: 700 !important; }
    h2 { font-weight: 600 !important; border-bottom: 1px solid #21262D; padding-bottom: 8px; }
    h3 { font-weight: 500 !important; color: #C9D1D9 !important; }

    /* Dividers */
    hr { border-color: #21262D !important; }

    /* DataFrames */
    .stDataFrame { border: 1px solid #21262D; border-radius: 8px; }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: 1px solid #2EA043;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
        font-size: 14px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #2EA043;
        border-color: #3FB950;
    }

    /* Alert boxes */
    .fraud-alert {
        background-color: #3D1B1B;
        border: 1px solid #CF2222;
        border-left: 4px solid #CF2222;
        border-radius: 8px;
        padding: 14px 18px;
        color: #FF7B7B;
        font-weight: 600;
        margin: 8px 0;
    }
    .safe-alert {
        background-color: #1A2D1A;
        border: 1px solid #238636;
        border-left: 4px solid #238636;
        border-radius: 8px;
        padding: 14px 18px;
        color: #56D364;
        font-weight: 600;
        margin: 8px 0;
    }
    .info-box {
        background-color: #1C2A3A;
        border: 1px solid #1F6FEB;
        border-left: 4px solid #1F6FEB;
        border-radius: 8px;
        padding: 14px 18px;
        color: #79C0FF;
        margin: 8px 0;
    }

    /* Stat pill */
    .stat-pill {
        display: inline-block;
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 13px;
        color: #8B949E;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section headers */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
        margin-bottom: 12px;
    }

    /* Progress indicator */
    .step-indicator {
        background-color: #21262D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #388BFD;
        font-size: 14px;
        color: #C9D1D9;
    }

    /* Table styling */
    .stTable { color: #E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fix_coord(val):
    """Fix malformed coordinate strings like '36.011.293' → 36.011293."""
    s = str(val)
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.replace('-', '').split('.')
    if len(parts) >= 2:
        integer_part = parts[0]
        decimal_part = ''.join(parts[1:])[:6]
        try:
            result = float(integer_part + '.' + decimal_part)
            return -result if '-' in s else result
        except ValueError:
            return 0.0
    return 0.0


def preprocess(df: pd.DataFrame, label_encoders=None, fit: bool = True):
    """
    Feature engineering pipeline. Returns (X_df, label_encoders).
    If fit=True, fits new encoders; otherwise uses the provided ones.
    """
    df = df.copy()

    # ── Datetime features ──
    df['trans_datetime'] = pd.to_datetime(
        df['trans_date_trans_time'], format='%d/%m/%Y %H:%M', errors='coerce'
    )
    df['trans_hour']    = df['trans_datetime'].dt.hour
    df['trans_day']     = df['trans_datetime'].dt.dayofweek
    df['trans_month']   = df['trans_datetime'].dt.month
    df['is_weekend']    = (df['trans_day'] >= 5).astype(int)
    df['is_night']      = ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 5)).astype(int)

    # ── Age ──
    df['dob_parsed'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
    df['age'] = (df['trans_datetime'] - df['dob_parsed']).dt.days // 365

    # ── Fix coordinates ──
    for col in ['merch_lat', 'merch_long', 'lat', 'long']:
        if col in df.columns:
            df[col + '_num'] = df[col].apply(fix_coord)

    # ── Distance between cardholder and merchant ──
    if all(c in df.columns for c in ['lat_num', 'long_num', 'merch_lat_num', 'merch_long_num']):
        df['dist'] = np.sqrt(
            (df['lat_num'] - df['merch_lat_num'])**2 +
            (df['long_num'] - df['merch_long_num'])**2
        )
    else:
        df['dist'] = 0

    # ── Categorical encoding ──
    cat_cols = ['category', 'gender']
    if label_encoders is None:
        label_encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            df[col + '_enc'] = 0
            continue
        if fit:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                df[col + '_enc'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
            else:
                df[col + '_enc'] = 0

    feature_cols = [
        'amt', 'trans_hour', 'trans_day', 'trans_month',
        'is_weekend', 'is_night', 'age', 'city_pop',
        'category_enc', 'gender_enc',
        'lat_num', 'long_num', 'merch_lat_num', 'merch_long_num', 'dist'
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available].fillna(0), label_encoders


def get_feature_names():
    return [
        'Jumlah Transaksi (amt)', 'Jam Transaksi', 'Hari (0=Sen)', 'Bulan',
        'Akhir Pekan', 'Jam Malam (22:00–05:00)', 'Usia Pemilik Kartu',
        'Populasi Kota', 'Kategori Merchant', 'Jenis Kelamin',
        'Lat. Kartu', 'Long. Kartu', 'Lat. Merchant', 'Long. Merchant',
        'Jarak Kartu–Merchant'
    ]


@st.cache_data(show_spinner=False)
def load_data():
    train_df = pd.read_csv('fraudTrain.csv')
    test_df  = pd.read_csv('fraudTest.csv')
    # Use sample if files are huge to keep training fast in demo
    if len(train_df) > 50_000:
        train_df = train_df.sample(n=50_000, random_state=42)
    if len(test_df) > 10_000:
        test_df = test_df.sample(n=10_000, random_state=42)
    return train_df, test_df


MODEL_PATH   = 'fraud_model.pkl'
ENCODER_PATH = 'label_encoders.pkl'


def _do_train(_train_df):
    """Core training logic — SMOTE + Random Forest."""
    X, le = preprocess(_train_df, fit=True)
    y = _train_df['is_fraud']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_res, y_res)
    return model, le


@st.cache_resource(show_spinner=False)
def load_or_train_model(_train_df):
    """
    Muat model dari pickle jika sudah ada.
    Jika belum, latih dari nol lalu simpan ke pickle.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f:
                label_encoders = pickle.load(f)
            return model, label_encoders, "loaded"
        except Exception:
            pass  # file rusak → latih ulang

    model, label_encoders = _do_train(_train_df)

    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoders, f)
    except Exception:
        pass  # filesystem read-only (misal Streamlit Cloud) → tetap lanjut

    return model, label_encoders, "trained"


def plot_confusion_matrix(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prediksi: Normal', 'Prediksi: Fraud'],
        y=['Aktual: Normal', 'Aktual: Fraud'],
        colorscale=[[0, '#0D1117'], [0.5, '#1F6FEB'], [1.0, '#388BFD']],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 18, "color": "white"},
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        xaxis=dict(tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
    )
    return fig


def plot_roc(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line=dict(color='#388BFD', width=2.5),
        name=f'ROC (AUC={auc_score:.3f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#30363D', width=1.5, dash='dash'),
        name='Random'
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8B949E'),
        xaxis=dict(gridcolor='#21262D', zeroline=False),
        yaxis=dict(gridcolor='#21262D', zeroline=False),
    )
    return fig


def plot_pr_curve(precision, recall, ap_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        line=dict(color='#56D364', width=2.5),
        name=f'PR (AP={ap_score:.3f})'
    ))
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8B949E'),
        xaxis=dict(gridcolor='#21262D', zeroline=False),
        yaxis=dict(gridcolor='#21262D', zeroline=False),
    )
    return fig


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    colors = ['#388BFD' if i == idx[-1] else '#1F6FEB' for i in range(len(idx))]

    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] if i < len(feature_names) else f'f{i}' for i in idx],
        orientation='h',
        marker_color=[colors[i] for i in range(len(idx))],
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis=dict(title='Importance', gridcolor='#21262D', zeroline=False),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


def plot_score_distribution(y_true, y_proba):
    df_plot = pd.DataFrame({'score': y_proba, 'label': y_true.map({0: 'Normal', 1: 'Fraud'})})
    fig = px.histogram(
        df_plot, x='score', color='label',
        barmode='overlay',
        color_discrete_map={'Normal': '#388BFD', 'Fraud': '#FF7B7B'},
        nbins=50,
        opacity=0.75,
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(title='Fraud Probability Score', gridcolor='#21262D'),
        yaxis=dict(title='Count', gridcolor='#21262D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def plot_category_fraud(df):
    fraud_by_cat = df.groupby('category').agg(
        total=('is_fraud', 'count'),
        fraud=('is_fraud', 'sum')
    ).reset_index()
    fraud_by_cat['fraud_rate'] = fraud_by_cat['fraud'] / fraud_by_cat['total'] * 100
    fraud_by_cat = fraud_by_cat.sort_values('fraud_rate', ascending=True)

    fig = go.Figure(go.Bar(
        x=fraud_by_cat['fraud_rate'],
        y=fraud_by_cat['category'],
        orientation='h',
        marker_color='#FF7B7B',
        text=fraud_by_cat['fraud_rate'].round(2).astype(str) + '%',
        textposition='outside',
        textfont=dict(color='#8B949E', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        xaxis=dict(title='Fraud Rate (%)', gridcolor='#21262D'),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


def plot_hourly_fraud(df):
    hourly = df.copy()
    hourly['hour'] = pd.to_datetime(
        hourly['trans_date_trans_time'], format='%d/%m/%Y %H:%M', errors='coerce'
    ).dt.hour
    h = hourly.groupby('hour').agg(total=('is_fraud', 'count'), fraud=('is_fraud', 'sum')).reset_index()
    h['fraud_rate'] = h['fraud'] / h['total'] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=h['hour'], y=h['total'], name='Total Transaksi', marker_color='#21262D'))
    fig.add_trace(go.Scatter(x=h['hour'], y=h['fraud_rate'], name='Fraud Rate (%)',
                             line=dict(color='#FF7B7B', width=2.5), yaxis='y2'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        yaxis=dict(title='Total Transaksi', gridcolor='#21262D'),
        yaxis2=dict(title='Fraud Rate (%)', overlaying='y', side='right', gridcolor='#21262D'),
        xaxis=dict(title='Jam (0–23)', gridcolor='#21262D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown('<div class="section-label">Navigasi</div>', unsafe_allow_html=True)

    page = st.radio(
        "Pilih halaman",
        ["📊 Dashboard", "🤖 Training & Evaluasi", "🔎 Prediksi Batch", "📂 Batch Predict CSV", "🧪 Simulasi Transaksi"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Konfigurasi Model</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Threshold Fraud", 0.1, 0.9, 0.5, 0.05,
        help="Transaksi dengan skor ≥ threshold akan diklasifikasikan sebagai fraud"
    )
    n_estimators = st.select_slider("Jumlah Pohon (n_estimators)", [50, 100, 150, 200], value=150)

    st.markdown("---")
    st.markdown('<div class="section-label">Tentang</div>', unsafe_allow_html=True)
    st.caption(
        "Model: Random Forest + SMOTE\n\n"
        "Dataset: Credit Card Transactions\n\n"
        "Features: 15 engineered features"
    )

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _find_csv(candidates: list[str]) -> str | None:
    """Return the first existing, non-empty path from a list of candidates."""
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def _read_csv_safe(path: str, label: str) -> pd.DataFrame:
    """
    Read a CSV with clear, user-visible error messages for every failure mode:
    - file missing
    - file empty (EmptyDataError)
    - file has no rows after the header
    - any other parse error
    """
    if not os.path.exists(path):
        st.error(f"❌ File **{label}** tidak ditemukan di path: `{path}`")
        st.info(
            "Pastikan file CSV sudah diupload ke repository GitHub kamu "
            "dan nama filenya persis sama (Linux bersifat case-sensitive)."
        )
        st.stop()

    if os.path.getsize(path) == 0:
        st.error(f"❌ File **{label}** berukuran 0 byte (kosong).")
        st.info("Upload ulang file CSV yang valid ke repositorimu.")
        st.stop()

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        st.error(f"❌ File **{label}** tidak memiliki data yang bisa dibaca (EmptyDataError).")
        st.info(
            "Kemungkinan penyebab:\n"
            "- File hanya berisi header tanpa baris data\n"
            "- File corrupt atau tidak berformat CSV\n"
            "- Git LFS tidak ter-pull dengan benar"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Gagal membaca **{label}**: `{e}`")
        st.stop()

    if df.empty:
        st.error(f"❌ File **{label}** berhasil dibaca tapi tidak ada baris data.")
        st.stop()

    return df


@st.cache_data(show_spinner=False)
def load_data_safe():
    """
    Load train & test CSVs with robust error handling.
    Tries several common filename variants (case-insensitive) so the app
    works whether the file is named fraudTrain.csv, fraudtrain.csv, etc.
    """
    train_candidates = [
        'fraudTrain.csv', 'fraudtrain.csv', 'FraudTrain.csv',
        'fraud_train.csv', 'train.csv',
    ]
    test_candidates = [
        'fraudTest.csv', 'fraudtest.csv', 'FraudTest.csv',
        'fraud_test.csv', 'test.csv',
    ]

    train_path = _find_csv(train_candidates)
    test_path  = _find_csv(test_candidates)

    if train_path is None:
        st.error("❌ File **fraudTrain.csv** tidak ditemukan.")
        st.markdown(
            "**Pastikan:**\n"
            "- File CSV sudah di-commit dan di-push ke GitHub\n"
            "- Nama file sesuai (coba: `fraudTrain.csv` atau `fraudtrain.csv`)\n"
            "- Jika file > 100 MB, gunakan [Git LFS](https://git-lfs.github.com/) "
            "atau simpan di Google Drive lalu load via URL"
        )
        st.stop()

    if test_path is None:
        st.error("❌ File **fraudTest.csv** tidak ditemukan.")
        st.stop()

    train_df = _read_csv_safe(train_path, "fraudTrain.csv")
    test_df  = _read_csv_safe(test_path,  "fraudTest.csv")

    # Validate required columns exist
    required_cols = {'amt', 'is_fraud', 'trans_date_trans_time', 'category', 'gender'}
    for df, name in [(train_df, 'fraudTrain'), (test_df, 'fraudTest')]:
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"❌ File **{name}** tidak memiliki kolom wajib: `{missing}`")
            st.stop()

    # Sample down if huge (keeps Streamlit Cloud within memory limits)
    if len(train_df) > 50_000:
        train_df = train_df.sample(n=50_000, random_state=42).reset_index(drop=True)
    if len(test_df) > 10_000:
        test_df = test_df.sample(n=10_000, random_state=42).reset_index(drop=True)

    return train_df, test_df


with st.spinner("⏳ Memuat data..."):
    train_df, test_df = load_data_safe()

with st.spinner("⏳ Memuat / melatih model..."):
    model, label_encoders, model_source = load_or_train_model(train_df)

# Tampilkan status model di sidebar
with st.sidebar:
    if model_source == "loaded":
        st.success("✅ Model dimuat dari cache (pickle)")
    else:
        st.info("🔄 Model baru dilatih & disimpan")

    # Tombol download model
    st.markdown("---")
    st.markdown('<div class="section-label">Download Model</div>', unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    import pickle as _pkl
    with col_dl1:
        model_bytes = _pkl.dumps(model)
        st.download_button(
            "⬇️ Model",
            data=model_bytes,
            file_name="fraud_model.pkl",
            mime="application/octet-stream",
            help="Download fraud_model.pkl",
            use_container_width=True,
        )
    with col_dl2:
        encoder_bytes = _pkl.dumps(label_encoders)
        st.download_button(
            "⬇️ Encoder",
            data=encoder_bytes,
            file_name="label_encoders.pkl",
            mime="application/octet-stream",
            help="Download label_encoders.pkl",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown("# 📊 Dashboard Analisis Data")
    st.markdown("Eksplorasi pola dan karakteristik transaksi pada dataset.")
    st.markdown("---")

    # Top stats
    total_train    = len(train_df)
    fraud_train    = train_df['is_fraud'].sum()
    normal_train   = total_train - fraud_train
    fraud_rate_pct = fraud_train / total_train * 100
    avg_amt        = train_df['amt'].mean()
    avg_fraud_amt  = train_df[train_df['is_fraud'] == 1]['amt'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transaksi", f"{total_train:,}")
    c2.metric("Transaksi Normal", f"{normal_train:,}", delta=f"{(normal_train/total_train*100):.1f}%")
    c3.metric("Transaksi Fraud", f"{fraud_train:,}", delta=f"{fraud_rate_pct:.2f}%", delta_color="inverse")
    c4.metric("Avg. Nominal", f"${avg_amt:,.0f}")
    c5.metric("Avg. Nominal Fraud", f"${avg_fraud_amt:,.0f}", delta=f"+${avg_fraud_amt - avg_amt:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Kelas")
        fig_pie = go.Figure(go.Pie(
            labels=['Normal', 'Fraud'],
            values=[normal_train, fraud_train],
            hole=0.55,
            marker_colors=['#388BFD', '#FF7B7B'],
            textinfo='percent+label',
            textfont_size=13,
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9',
            margin=dict(l=0, r=0, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### Fraud Rate per Kategori Merchant")
        st.plotly_chart(plot_category_fraud(train_df), use_container_width=True)

    st.markdown("### Pola Fraud per Jam Transaksi")
    st.plotly_chart(plot_hourly_fraud(train_df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Distribusi Nominal Transaksi")
        fig_amt = px.histogram(
            train_df[train_df['amt'] < 500],
            x='amt', color=train_df[train_df['amt'] < 500]['is_fraud'].map({0: 'Normal', 1: 'Fraud'}),
            color_discrete_map={'Normal': '#388BFD', 'Fraud': '#FF7B7B'},
            nbins=60, opacity=0.75, barmode='overlay',
        )
        fig_amt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0D1117',
            font_color='#C9D1D9', margin=dict(l=10, r=10, t=10, b=10),
            height=280, legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D'),
        )
        st.plotly_chart(fig_amt, use_container_width=True)

    with col4:
        st.markdown("### Fraud Rate per Jenis Kelamin")
        gender_stats = train_df.groupby('gender').agg(
            fraud=('is_fraud', 'sum'), total=('is_fraud', 'count')
        ).reset_index()
        gender_stats['rate'] = gender_stats['fraud'] / gender_stats['total'] * 100
        fig_g = px.bar(
            gender_stats, x='gender', y='rate',
            text=gender_stats['rate'].round(2).astype(str) + '%',
            color_discrete_sequence=['#388BFD'],
        )
        fig_g.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0D1117',
            font_color='#C9D1D9', margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(title='Jenis Kelamin', gridcolor='#21262D'),
            yaxis=dict(title='Fraud Rate (%)', gridcolor='#21262D'),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")
    st.markdown("### Sample Data (Training)")
    st.dataframe(
        train_df[['trans_date_trans_time', 'merchant', 'category', 'amt', 'gender', 'city', 'state', 'is_fraud']].head(100),
        use_container_width=True, height=280
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING & EVALUASI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Training & Evaluasi":
    st.markdown("# 🤖 Training & Evaluasi Model")
    st.markdown("Performa model Random Forest dengan oversampling SMOTE pada data test.")
    st.markdown("---")

    # Prepare test data & predict
    with st.spinner("Mengevaluasi model pada data test..."):
        X_test, _ = preprocess(test_df, label_encoders=label_encoders, fit=False)
        y_test    = test_df['is_fraud']
        y_proba   = model.predict_proba(X_test)[:, 1]
        y_pred    = (y_proba >= threshold).astype(int)

        roc_auc   = roc_auc_score(y_test, y_proba)
        ap_score  = average_precision_score(y_test, y_proba)
        cm        = confusion_matrix(y_test, y_pred)
        cr        = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area Under Curve — semakin dekat 1 semakin baik")
    col2.metric("Average Precision", f"{ap_score:.4f}", help="Ringkasan precision-recall curve")
    col3.metric("Precision (Fraud)", f"{cr['1']['precision']:.2%}")
    col4.metric("Recall (Fraud)", f"{cr['1']['recall']:.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("True Positive", f"{tp:,}", help="Fraud terdeteksi dengan benar")
    col6.metric("False Positive", f"{fp:,}", delta=f"−{fp}", delta_color="inverse", help="Normal diklasifikasikan sebagai fraud")
    col7.metric("False Negative", f"{fn:,}", delta=f"−{fn}", delta_color="inverse", help="Fraud tidak terdeteksi")
    col8.metric("True Negative", f"{tn:,}", help="Normal terdeteksi dengan benar")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### ROC Curve")
        st.plotly_chart(plot_roc(fpr, tpr, roc_auc), use_container_width=True)

    with col_r:
        st.markdown("### Precision-Recall Curve")
        st.plotly_chart(plot_pr_curve(prec, rec, ap_score), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    with col_b:
        st.markdown("### Distribusi Skor Probabilitas")
        st.plotly_chart(plot_score_distribution(y_test, y_proba), use_container_width=True)

    st.markdown("### Feature Importance")
    feat_names = get_feature_names()[:len(model.feature_importances_)]
    st.plotly_chart(plot_feature_importance(model, feat_names), use_container_width=True)

    st.markdown("---")
    st.markdown("### Pipeline Model")
    steps = [
        ("1", "Feature Engineering", "Datetime parsing, age extraction, koordinat normalisasi, distance calculation, label encoding"),
        ("2", "SMOTE Oversampling", "Minority class (Fraud) di-oversample hingga seimbang dengan class Normal"),
        ("3", "Random Forest Classifier", f"n_estimators=150, max_depth=12, class_weight='balanced', n_jobs=-1"),
        ("4", "Threshold Tuning", f"Threshold saat ini: {threshold:.2f} — sesuaikan di sidebar untuk trade-off precision/recall"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-indicator">
            <strong>Step {num} — {title}</strong><br>
            <span style="color:#8B949E; font-size:13px">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI BATCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Prediksi Batch":
    st.markdown("# 🔎 Prediksi Batch pada Data Test")
    st.markdown("Jalankan prediksi terhadap seluruh data test dan unduh hasilnya.")
    st.markdown("---")

    X_test, _ = preprocess(test_df, label_encoders=label_encoders, fit=False)
    y_proba   = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)

    result_df = test_df[['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
                          'first', 'last', 'city', 'state', 'is_fraud']].copy().reset_index(drop=True)
    result_df['fraud_score']     = y_proba.round(4)
    result_df['prediksi']        = y_pred
    result_df['prediksi_label']  = result_df['prediksi'].map({0: '✅ Normal', 1: '🚨 FRAUD'})
    result_df['benar']           = (result_df['prediksi'] == result_df['is_fraud']).map({True: '✓', False: '✗'})

    # Summary
    n_fraud_pred = y_pred.sum()
    n_total      = len(y_pred)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris Diprediksi", f"{n_total:,}")
    col2.metric("Prediksi Fraud", f"{n_fraud_pred:,}", delta=f"{n_fraud_pred/n_total*100:.2f}%", delta_color="inverse")
    col3.metric("Prediksi Normal", f"{n_total - n_fraud_pred:,}")
    col4.metric("Threshold", f"{threshold:.2f}")

    st.markdown("---")

    # Filter
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_view = st.selectbox("Filter tampilan", ["Semua", "Hanya Fraud Terdeteksi", "Hanya Normal", "Salah Klasifikasi"])
    with col_f2:
        score_min, score_max = st.slider("Rentang Fraud Score", 0.0, 1.0, (0.0, 1.0), 0.01)

    display_df = result_df.copy()
    if filter_view == "Hanya Fraud Terdeteksi":
        display_df = display_df[display_df['prediksi'] == 1]
    elif filter_view == "Hanya Normal":
        display_df = display_df[display_df['prediksi'] == 0]
    elif filter_view == "Salah Klasifikasi":
        display_df = display_df[display_df['benar'] == '✗']

    display_df = display_df[(display_df['fraud_score'] >= score_min) & (display_df['fraud_score'] <= score_max)]

    st.dataframe(
        display_df[['trans_date_trans_time', 'merchant', 'category', 'amt',
                     'first', 'last', 'fraud_score', 'prediksi_label', 'benar']],
        use_container_width=True, height=460
    )
    st.caption(f"Menampilkan {len(display_df):,} dari {n_total:,} baris")

    # Download
    csv_out = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Unduh Hasil Prediksi (CSV)",
        data=csv_out,
        file_name='hasil_prediksi_fraud.csv',
        mime='text/csv',
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT CSV
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Batch Predict CSV":
    st.markdown("# 📂 Batch Predict dari File CSV")
    st.markdown("Upload file CSV berisi data transaksi, lalu jalankan prediksi fraud secara massal.")
    st.markdown("---")

    # ── Kolom wajib yang harus ada di CSV ─────────────────────────────────────
    REQUIRED_COLS = {
        'trans_date_trans_time', 'amt', 'category', 'gender',
        'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'dob'
    }
    OPTIONAL_DISPLAY_COLS = ['merchant', 'cc_num', 'first', 'last', 'city', 'state', 'is_fraud']

    # ── Template download ──────────────────────────────────────────────────────
    with st.expander("📋 Panduan Format CSV & Download Template", expanded=False):
        st.markdown("""
        File CSV yang diupload **wajib** memiliki kolom berikut:

        | Kolom | Tipe | Contoh | Keterangan |
        |---|---|---|---|
        | `trans_date_trans_time` | string | `01/06/2024 14:30` | Format: `DD/MM/YYYY HH:MM` |
        | `amt` | float | `125.50` | Nominal transaksi (USD) |
        | `category` | string | `grocery_pos` | Kategori merchant |
        | `gender` | string | `M` / `F` | Jenis kelamin pemilik kartu |
        | `lat` | float/string | `37.7749` | Latitude lokasi kartu |
        | `long` | float/string | `-122.4194` | Longitude lokasi kartu |
        | `merch_lat` | float/string | `37.8000` | Latitude merchant |
        | `merch_long` | float/string | `-122.4000` | Longitude merchant |
        | `city_pop` | int | `50000` | Populasi kota |
        | `dob` | string | `15/08/1990` | Tanggal lahir, format: `DD/MM/YYYY` |

        Kolom opsional (jika ada, akan ditampilkan di hasil):
        `merchant`, `cc_num`, `first`, `last`, `city`, `state`, `is_fraud`
        """)

        # Buat template CSV
        template_data = {
            'trans_date_trans_time': ['01/06/2024 14:30', '02/06/2024 02:15', '03/06/2024 10:00'],
            'amt': [125.50, 1850.00, 45.20],
            'category': ['grocery_pos', 'shopping_net', 'gas_transport'],
            'gender': ['F', 'M', 'F'],
            'lat': [37.7749, 40.7128, 34.0522],
            'long': [-122.4194, -74.0060, -118.2437],
            'merch_lat': [37.8000, 42.3601, 33.9000],
            'merch_long': [-122.4000, -71.0589, -118.0000],
            'city_pop': [50000, 800000, 200000],
            'dob': ['15/08/1990', '22/03/1985', '07/11/1978'],
            'merchant': ['Walmart', 'Amazon', 'Shell'],
            'first': ['Budi', 'Siti', 'Ahmad'],
            'last': ['Santoso', 'Rahayu', 'Fauzi'],
            'city': ['San Francisco', 'New York', 'Los Angeles'],
            'state': ['CA', 'NY', 'CA'],
        }
        template_df = pd.DataFrame(template_data)
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Template CSV",
            data=template_csv,
            file_name='template_batch_predict.csv',
            mime='text/csv',
        )

    st.markdown("---")

    # ── Upload area ────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload file CSV transaksi",
        type=["csv"],
        help="Maksimal 200 MB. Pastikan format kolom sesuai template."
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ Belum ada file yang diupload. Upload file CSV di atas untuk memulai prediksi batch.<br>
            Gunakan tombol <strong>Download Template CSV</strong> di atas untuk mendapatkan contoh format yang benar.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Baca CSV ───────────────────────────────────────────────────────────
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file CSV: `{e}`")
            st.stop()

        if upload_df.empty:
            st.error("❌ File CSV tidak memiliki data (kosong).")
            st.stop()

        # ── Validasi kolom ─────────────────────────────────────────────────────
        missing_cols = REQUIRED_COLS - set(upload_df.columns)
        if missing_cols:
            st.error(f"❌ Kolom wajib tidak ditemukan: `{sorted(missing_cols)}`")
            st.markdown("""
            <div class="info-box">
                ℹ️ Pastikan file CSV memiliki semua kolom wajib.<br>
                Download template di atas sebagai referensi format yang benar.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ── Info file & preview ────────────────────────────────────────────────
        col_fi1, col_fi2, col_fi3, col_fi4 = st.columns(4)
        col_fi1.metric("Total Baris", f"{len(upload_df):,}")
        col_fi2.metric("Total Kolom", f"{len(upload_df.columns)}")
        col_fi3.metric("Ukuran File", f"{uploaded_file.size / 1024:.1f} KB")
        has_label = 'is_fraud' in upload_df.columns
        col_fi4.metric("Label Tersedia", "✅ Ya" if has_label else "—")

        with st.expander("🔍 Preview Data (10 baris pertama)", expanded=True):
            st.dataframe(upload_df.head(10), use_container_width=True)

        st.markdown("---")

        # ── Jalankan prediksi ──────────────────────────────────────────────────
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_predict = st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True)
        with col_info:
            st.caption(f"Model aktif: Random Forest + SMOTE  |  Threshold: {threshold:.2f}  |  {len(upload_df):,} baris akan diproses")

        if run_predict:
            with st.spinner(f"⏳ Memproses {len(upload_df):,} transaksi..."):
                try:
                    X_upload, _ = preprocess(
                        upload_df,
                        label_encoders=label_encoders,
                        fit=False
                    )
                    y_proba_upload = model.predict_proba(X_upload)[:, 1]
                    y_pred_upload  = (y_proba_upload >= threshold).astype(int)
                except Exception as e:
                    st.error(f"❌ Prediksi gagal: `{e}`")
                    st.stop()

            # ── Susun hasil ────────────────────────────────────────────────────
            # Kolom tampilan: ambil kolom opsional yang tersedia
            disp_cols = [c for c in OPTIONAL_DISPLAY_COLS if c in upload_df.columns and c != 'is_fraud']
            base_cols = ['trans_date_trans_time', 'amt', 'category', 'gender']
            show_cols = base_cols + [c for c in disp_cols if c not in base_cols]

            result_upload = upload_df[show_cols].copy().reset_index(drop=True)
            result_upload['fraud_score']    = y_proba_upload.round(4)
            result_upload['prediksi']       = y_pred_upload
            result_upload['prediksi_label'] = pd.Series(y_pred_upload).map({0: '✅ Normal', 1: '🚨 FRAUD'})
            result_upload['risiko']         = pd.cut(
                y_proba_upload,
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['🟢 Rendah', '🟡 Sedang', '🟠 Tinggi', '🔴 Kritis']
            )

            if has_label:
                result_upload['aktual']     = upload_df['is_fraud'].map({0: 'Normal', 1: 'Fraud'}).values
                result_upload['benar']      = (y_pred_upload == upload_df['is_fraud'].values).map({True: '✓', False: '✗'})

            # ── Ringkasan metrik ───────────────────────────────────────────────
            n_total_up   = len(y_pred_upload)
            n_fraud_up   = y_pred_upload.sum()
            n_normal_up  = n_total_up - n_fraud_up
            avg_score    = y_proba_upload.mean()
            max_score    = y_proba_upload.max()

            st.markdown("### 📊 Ringkasan Hasil Prediksi")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Transaksi", f"{n_total_up:,}")
            c2.metric("Terdeteksi Normal", f"{n_normal_up:,}", delta=f"{n_normal_up/n_total_up*100:.1f}%")
            c3.metric("Terdeteksi Fraud", f"{n_fraud_up:,}", delta=f"{n_fraud_up/n_total_up*100:.2f}%", delta_color="inverse")
            c4.metric("Avg. Fraud Score", f"{avg_score:.3f}")
            c5.metric("Max Fraud Score", f"{max_score:.3f}")

            # ── Visualisasi ringkasan ──────────────────────────────────────────
            col_v1, col_v2, col_v3 = st.columns(3)

            with col_v1:
                st.markdown("#### Distribusi Prediksi")
                fig_pie_up = go.Figure(go.Pie(
                    labels=['Normal', 'Fraud'],
                    values=[n_normal_up, n_fraud_up],
                    hole=0.55,
                    marker_colors=['#388BFD', '#FF7B7B'],
                    textinfo='percent+value',
                    textfont_size=12,
                ))
                fig_pie_up.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#C9D1D9',
                    margin=dict(l=0, r=0, t=10, b=10),
                    height=240,
                    showlegend=True,
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_pie_up, use_container_width=True)

            with col_v2:
                st.markdown("#### Distribusi Fraud Score")
                fig_hist_up = go.Figure()
                fig_hist_up.add_trace(go.Histogram(
                    x=y_proba_upload[y_pred_upload == 0],
                    name='Normal', nbinsx=30,
                    marker_color='#388BFD', opacity=0.75,
                ))
                fig_hist_up.add_trace(go.Histogram(
                    x=y_proba_upload[y_pred_upload == 1],
                    name='Fraud', nbinsx=30,
                    marker_color='#FF7B7B', opacity=0.75,
                ))
                fig_hist_up.add_vline(
                    x=threshold, line_dash="dash",
                    line_color="#FFD700", line_width=2,
                    annotation_text=f"Threshold={threshold}",
                    annotation_font_color="#FFD700",
                )
                fig_hist_up.update_layout(
                    barmode='overlay',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0D1117',
                    font_color='#C9D1D9',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    xaxis=dict(title='Fraud Score', gridcolor='#21262D'),
                    yaxis=dict(title='Count', gridcolor='#21262D'),
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_hist_up, use_container_width=True)

            with col_v3:
                st.markdown("#### Distribusi Level Risiko")
                risk_counts = result_upload['risiko'].value_counts()
                risk_order  = ['🟢 Rendah', '🟡 Sedang', '🟠 Tinggi', '🔴 Kritis']
                risk_colors = ['#56D364', '#E3B341', '#F0883E', '#FF7B7B']
                risk_vals   = [risk_counts.get(r, 0) for r in risk_order]

                fig_risk = go.Figure(go.Bar(
                    x=risk_order, y=risk_vals,
                    marker_color=risk_colors,
                    text=risk_vals,
                    textposition='outside',
                    textfont=dict(color='#C9D1D9'),
                ))
                fig_risk.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0D1117',
                    font_color='#C9D1D9',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    xaxis=dict(gridcolor='#21262D'),
                    yaxis=dict(title='Jumlah', gridcolor='#21262D'),
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # ── Jika label tersedia → tampilkan evaluasi ───────────────────────
            if has_label:
                from sklearn.metrics import classification_report as _cr, confusion_matrix as _cm
                _y_true = upload_df['is_fraud'].values
                _cr_dict = _cr(_y_true, y_pred_upload, output_dict=True, zero_division=0)
                _cm_vals = _cm(_y_true, y_pred_upload)

                st.markdown("### 🎯 Evaluasi vs Label Aktual")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy",         f"{_cr_dict['accuracy']:.2%}")
                e2.metric("Precision (Fraud)", f"{_cr_dict.get('1', {}).get('precision', 0):.2%}")
                e3.metric("Recall (Fraud)",    f"{_cr_dict.get('1', {}).get('recall', 0):.2%}")
                e4.metric("F1-Score (Fraud)",  f"{_cr_dict.get('1', {}).get('f1-score', 0):.2%}")

                col_cm, col_detail = st.columns([1, 2])
                with col_cm:
                    st.markdown("#### Confusion Matrix")
                    tn_u, fp_u, fn_u, tp_u = _cm_vals.ravel()
                    fig_cm_u = go.Figure(data=go.Heatmap(
                        z=_cm_vals,
                        x=['Prediksi Normal', 'Prediksi Fraud'],
                        y=['Aktual Normal', 'Aktual Fraud'],
                        colorscale=[[0,'#0D1117'],[0.5,'#1F6FEB'],[1,'#388BFD']],
                        text=_cm_vals,
                        texttemplate='<b>%{text}</b>',
                        textfont={"size": 16, "color": "white"},
                        showscale=False,
                    ))
                    fig_cm_u.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#C9D1D9',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=240,
                    )
                    st.plotly_chart(fig_cm_u, use_container_width=True)

                with col_detail:
                    st.markdown("#### Detail Klasifikasi")
                    cr_rows = []
                    for label_key, label_name in [('0','Normal'), ('1','Fraud'), ('macro avg','Macro Avg')]:
                        row = _cr_dict.get(label_key, {})
                        if row:
                            cr_rows.append({
                                'Kelas'    : label_name,
                                'Precision': f"{row.get('precision',0):.2%}",
                                'Recall'   : f"{row.get('recall',0):.2%}",
                                'F1-Score' : f"{row.get('f1-score',0):.2%}",
                                'Support'  : f"{int(row.get('support',0)):,}",
                            })
                    st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

            # ── Tabel hasil dengan filter ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Tabel Hasil Prediksi")

            col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
            with col_f1:
                filter_pred = st.selectbox(
                    "Filter Prediksi",
                    ["Semua", "Hanya Fraud", "Hanya Normal"],
                    key="bp_filter_pred"
                )
            with col_f2:
                filter_risk = st.selectbox(
                    "Filter Level Risiko",
                    ["Semua", "🔴 Kritis", "🟠 Tinggi", "🟡 Sedang", "🟢 Rendah"],
                    key="bp_filter_risk"
                )
            with col_f3:
                score_range = st.slider(
                    "Rentang Fraud Score",
                    0.0, 1.0, (0.0, 1.0), 0.01,
                    key="bp_score_range"
                )

            tbl = result_upload.copy()
            if filter_pred == "Hanya Fraud":
                tbl = tbl[tbl['prediksi'] == 1]
            elif filter_pred == "Hanya Normal":
                tbl = tbl[tbl['prediksi'] == 0]
            if filter_risk != "Semua":
                tbl = tbl[tbl['risiko'] == filter_risk]
            tbl = tbl[(tbl['fraud_score'] >= score_range[0]) & (tbl['fraud_score'] <= score_range[1])]

            # Kolom yang ditampilkan di tabel
            tbl_show = ['trans_date_trans_time', 'amt', 'category', 'gender',
                        'fraud_score', 'prediksi_label', 'risiko']
            if has_label:
                tbl_show += ['aktual', 'benar']
            tbl_show = [c for c in tbl_show if c in tbl.columns]

            st.dataframe(tbl[tbl_show], use_container_width=True, height=420)
            st.caption(f"Menampilkan {len(tbl):,} dari {n_total_up:,} baris")

            # ── Download hasil ─────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### ⬇️ Download Hasil")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            # Semua hasil
            csv_all = result_upload.to_csv(index=False).encode('utf-8')
            col_dl1.download_button(
                "⬇️ Download Semua Hasil (.csv)",
                data=csv_all,
                file_name='batch_predict_semua.csv',
                mime='text/csv',
                use_container_width=True,
            )

            # Hanya fraud
            fraud_only = result_upload[result_upload['prediksi'] == 1]
            csv_fraud  = fraud_only.to_csv(index=False).encode('utf-8')
            col_dl2.download_button(
                f"⬇️ Download Fraud Saja ({n_fraud_up:,} baris)",
                data=csv_fraud,
                file_name='batch_predict_fraud_only.csv',
                mime='text/csv',
                use_container_width=True,
                disabled=(n_fraud_up == 0),
            )

            # Hasil yang difilter (tabel aktif)
            csv_filtered = tbl[tbl_show].to_csv(index=False).encode('utf-8')
            col_dl3.download_button(
                f"⬇️ Download Hasil Terfilter ({len(tbl):,} baris)",
                data=csv_filtered,
                file_name='batch_predict_filtered.csv',
                mime='text/csv',
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULASI TRANSAKSI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Simulasi Transaksi":
    st.markdown("# 🧪 Simulasi Transaksi")
    st.markdown("Masukkan detail transaksi secara manual dan lihat apakah model mendeteksinya sebagai fraud.")
    st.markdown("---")

    all_categories = sorted(train_df['category'].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Informasi Transaksi")
        amt          = st.number_input("Nominal Transaksi ($)", min_value=0.01, max_value=100000.0, value=150.0, step=0.01)
        category     = st.selectbox("Kategori Merchant", all_categories)
        trans_hour   = st.slider("Jam Transaksi", 0, 23, 14)
        trans_day    = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        trans_month  = st.slider("Bulan", 1, 12, 6)

    with col2:
        st.markdown("#### Informasi Pemilik Kartu")
        age          = st.slider("Usia Pemilik Kartu", 18, 90, 35)
        gender       = st.radio("Jenis Kelamin", ["M", "F"], horizontal=True)
        city_pop     = st.number_input("Populasi Kota", min_value=100, max_value=5_000_000, value=50000, step=100)
        lat          = st.number_input("Latitude Kartu", value=37.5, format="%.4f")
        long_val     = st.number_input("Longitude Kartu", value=-95.5, format="%.4f")
        merch_lat    = st.number_input("Latitude Merchant", value=38.0, format="%.4f")
        merch_long   = st.number_input("Longitude Merchant", value=-96.0, format="%.4f")

    day_map  = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
    day_num  = day_map[trans_day]
    is_wknd  = int(day_num >= 5)
    is_night = int(trans_hour >= 22 or trans_hour <= 5)
    dist_val = np.sqrt((lat - merch_lat)**2 + (long_val - merch_long)**2)

    # Encode category & gender
    cat_le   = label_encoders.get('category')
    gen_le   = label_encoders.get('gender')
    cat_enc  = cat_le.transform([category])[0] if cat_le and category in cat_le.classes_ else 0
    gen_enc  = gen_le.transform([gender])[0]   if gen_le and gender in gen_le.classes_   else 0

    feature_vector = np.array([[
        amt, trans_hour, day_num, trans_month,
        is_wknd, is_night, age, city_pop,
        cat_enc, gen_enc,
        lat, long_val, merch_lat, merch_long, dist_val
    ]])

    st.markdown("---")
    if st.button("🔍 Analisis Transaksi Ini"):
        proba   = model.predict_proba(feature_vector)[0][1]
        is_fraud_pred = proba >= threshold

        st.markdown("### Hasil Analisis")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Fraud Score", f"{proba:.4f}")
        col_res2.metric("Threshold", f"{threshold:.2f}")
        col_res3.metric("Status", "🚨 FRAUD" if is_fraud_pred else "✅ NORMAL")

        if is_fraud_pred:
            st.markdown(f"""
            <div class="fraud-alert">
                🚨 PERINGATAN: Transaksi ini terdeteksi sebagai <strong>FRAUD</strong><br>
                Skor probabilitas: <strong>{proba:.2%}</strong> — melebihi threshold {threshold:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                ✅ Transaksi ini diprediksi <strong>NORMAL</strong><br>
                Skor probabilitas: <strong>{proba:.2%}</strong> — di bawah threshold {threshold:.2f}
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': '%', 'font': {'color': '#FF7B7B' if is_fraud_pred else '#56D364', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8B949E'},
                'bar': {'color': '#FF7B7B' if is_fraud_pred else '#56D364'},
                'bgcolor': '#21262D',
                'bordercolor': '#30363D',
                'steps': [
                    {'range': [0, threshold * 100], 'color': '#1A2D1A'},
                    {'range': [threshold * 100, 100], 'color': '#3D1B1B'},
                ],
                'threshold': {
                    'line': {'color': '#FFD700', 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9',
            margin=dict(l=20, r=20, t=30, b=20),
            height=240,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature context
        st.markdown("#### Ringkasan Fitur yang Digunakan")
        feat_display = {
            "Nominal": f"${amt:,.2f}",
            "Kategori": category,
            "Jam": f"{trans_hour}:00",
            "Hari": trans_day,
            "Akhir Pekan": "Ya" if is_wknd else "Tidak",
            "Jam Malam": "Ya" if is_night else "Tidak",
            "Usia": f"{age} tahun",
            "Pop. Kota": f"{city_pop:,}",
            "Jarak ke Merchant": f"{dist_val:.4f}°",
        }
        pills_html = "".join([f'<span class="stat-pill">{k}: {v}</span>' for k, v in feat_display.items()])
        st.markdown(f'<div style="margin-top:8px">{pills_html}</div>', unsafe_allow_html=True)
