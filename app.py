import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import io

st.set_page_config(page_title="DermSight AI", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border-radius: 16px;
        padding: 40px 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #F8FAFC;
        letter-spacing: 0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: #CBD5E1;
        margin-bottom: 0;
    }

    .stButton>button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
        color: white;
    }

    .info-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        border-left: 6px solid #3B82F6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: #E2E8F0;
        margin: 30px 0;
    }

    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        background: white;
    }
    </style>
""", unsafe_allow_html=True)

WEIGHTS_PATH = 'model_weights.h5' 
IMG_SIZE = 384
NUM_CLASSES = 6
CONFIDENCE_THRESHOLD = 0.50
STRIDE = 100 

class_names = ['Acne', 'Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Normal Skin', 'Squamous Cell Carcinoma']

medical_info = {
    'Melanoma': {
        'risk': 'SANGAT TINGGI',
        'risk_color': '#FEE2E2', 'risk_text': '#DC2626',
        'description': 'Jenis kanker kulit yang sangat agresif. Berkembang dari sel melanosit penghasil pigmen.',
        'advice': 'Segera konsultasikan dengan Dokter Spesialis Kulit untuk prosedur biopsi. Membutuhkan penanganan medis segera.'
    },
    'Squamous Cell Carcinoma': {
        'risk': 'MENENGAH-TINGGI',
        'risk_color': '#FFEDD5', 'risk_text': '#C2410C',
        'description': 'Kanker kulit yang berkembang di lapisan epidermis. Sering muncul di area yang terpapar sinar matahari.',
        'advice': 'Harus diperiksa oleh dokter. Tingkat kesembuhan sangat baik jika dideteksi dan diangkat sejak dini.'
    },
    'Basal Cell Carcinoma': {
        'risk': 'MENENGAH',
        'risk_color': '#FEF9C3', 'risk_text': '#A16207',
        'description': 'Kanker kulit paling umum. Pertumbuhannya lambat dan jarang menyebar, namun dapat merusak jaringan lokal.',
        'advice': 'Konsultasikan dengan dokter spesialis kulit untuk prosedur pengangkatan lesi.'
    },
    'Acne': {
        'risk': 'RENDAH',
        'risk_color': '#D1FAE5', 'risk_text': '#047857',
        'description': 'Kondisi peradangan pada kelenjar minyak kulit. Termasuk komedo, papula, atau pustula.',
        'advice': 'Jaga kebersihan kulit. Gunakan produk perawatan yang sesuai atau konsultasikan dengan dokter untuk jerawat meradang.'
    },
    'Nevus': {
        'risk': 'AMAN',
        'risk_color': '#E0F2FE', 'risk_text': '#0369A1',
        'description': 'Tahi lalat jinak. Kumpulan sel melanosit normal yang tidak memiliki potensi bahaya.',
        'advice': 'Tidak memerlukan tindakan medis. Disarankan pemantauan mandiri berkala terhadap perubahan ukuran atau warna.'
    },
    'Normal Skin': {
        'risk': 'SANGAT AMAN',
        'risk_color': '#DBEAFE', 'risk_text': '#1D4ED8',
        'description': 'Jaringan kulit terpantau sehat tanpa adanya kelainan lesi atau pigmentasi abnormal.',
        'advice': 'Pertahankan rutinitas kebersihan kulit dan gunakan tabir surya secara teratur saat beraktivitas.'
    }
}

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"File '{WEIGHTS_PATH}' tidak ditemukan.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(WEIGHTS_PATH)
        return model
    except Exception as e:
        try:
            base_model = tf.keras.applications.EfficientNetV2S(
                include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
            ])
            model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
            return model
        except Exception as e_final:
            st.error(f"Gagal memuat model: {e_final}")
            st.stop()

def process_results(preds_batch, coords, boxes, confidences, class_ids):
    for i, preds in enumerate(preds_batch):
        x, y = coords[i]
        p_bcc, p_mel, p_scc = preds[1], preds[2], preds[5]
        
        if p_mel > 0.30 or p_scc > 0.30 or p_bcc > 0.30:
            suspects = []
            if p_mel > 0.20: suspects.append((p_mel, 2, "Melanoma"))
            if p_scc > 0.20: suspects.append((p_scc, 5, "SCC"))
            if p_bcc > 0.20: suspects.append((p_bcc, 1, "BCC"))
            suspects.sort(key=lambda x: x[0], reverse=True)
            
            if len(suspects) >= 2:
                boxes.append([x, y, IMG_SIZE, IMG_SIZE])
                confidences.append(float(suspects[0][0]))
                class_ids.append(f"{suspects[0][2]} / {suspects[1][2]}")
            elif len(suspects) == 1:
                if suspects[0][0] > 0.30: 
                    boxes.append([x, y, IMG_SIZE, IMG_SIZE])
                    confidences.append(float(suspects[0][0]))
                    class_ids.append(suspects[0][1])
        else:
            class_id = np.argmax(preds)
            confidence = preds[class_id]
            label = class_names[class_id]
            thresh = 0.35 if label == 'Acne' else CONFIDENCE_THRESHOLD
            
            if label != 'Normal Skin' and confidence > thresh:
                boxes.append([x, y, IMG_SIZE, IMG_SIZE])
                confidences.append(float(confidence))
                class_ids.append(class_id)

def predict_image(image_file, model):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = original_img.shape
    if h < IMG_SIZE or w < IMG_SIZE:
        scale = max(IMG_SIZE/h, IMG_SIZE/w)
        original_img = cv2.resize(original_img, (int(w*scale)+1, int(h*scale)+1))
        h, w, _ = original_img.shape

    progress_bar = st.progress(0, text="Memproses analisis citra...")
    batch_img, batch_xy, boxes, confs, ids = [], [], [], [], []
    y_range = range(0, h - IMG_SIZE + 1, STRIDE)
    x_range = range(0, w - IMG_SIZE + 1, STRIDE)
    if not y_range: y_range = [0]
    if not x_range: x_range = [0]
    
    total = len(y_range) * len(x_range)
    processed = 0

    for y in y_range:
        for x in x_range:
            if y+IMG_SIZE > h or x+IMG_SIZE > w: continue
            patch = original_img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            input_arr = tf.keras.applications.efficientnet_v2.preprocess_input(cv2.resize(patch, (IMG_SIZE, IMG_SIZE)))
            batch_img.append(input_arr); batch_xy.append((x, y))
            if len(batch_img) == 32:
                preds = model.predict(np.array(batch_img), verbose=0)
                process_results(preds, batch_xy, boxes, confs, ids)
                batch_img, batch_xy = [], []
            processed += 1
            if total > 0: progress_bar.progress(min(processed/total, 1.0))
    if batch_img:
        preds = model.predict(np.array(batch_img), verbose=0)
        process_results(preds, batch_xy, boxes, confs, ids)
    progress_bar.empty()

    indices = cv2.dnn.NMSBoxes(boxes, confs, CONFIDENCE_THRESHOLD, 0.3)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(original_img)
    
    highest_confidence_per_class = {}
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            score = confs[i]
            raw_id = ids[i]
            if isinstance(raw_id, str):
                label = raw_id; box_color = '#F59E0B'; font_color = 'black'
            else:
                label = class_names[raw_id]
                if label == 'Melanoma': box_color = '#DC2626'; font_color = 'white'
                elif label == 'Squamous Cell Carcinoma': box_color = '#EA580C'; font_color = 'white'
                elif label == 'Basal Cell Carcinoma': box_color = '#CA8A04'; font_color = 'white'
                elif label == 'Acne':
                    box_color = '#10B981'; font_color = 'white'
                    shrink = 0.5
                    new_w, new_h = int(w_box*shrink), int(h_box*shrink)
                    x += (w_box-new_w)//2; y += (h_box-new_h)//2
                    w_box, h_box = new_w, new_h
                else: box_color = '#3B82F6'; font_color = 'white'
                
            ax.add_patch(plt.Rectangle((x, y), w_box, h_box, fill=False, color=box_color, linewidth=3))
            ax.text(x, y-10, f"{label} ({score*100:.0f}%)", color=font_color, bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.4'), fontsize=10, fontweight='bold')
            
            if label not in highest_confidence_per_class or score > highest_confidence_per_class[label]:
                highest_confidence_per_class[label] = score
                
    detected_data = []
    for label, score in highest_confidence_per_class.items():
        detected_data.append({'Kondisi': label, 'Akurasi AI': f"{score*100:.1f}%"})
    
    detected_data = sorted(detected_data, key=lambda x: float(x['Akurasi AI'].strip('%')), reverse=True)
            
    ax.axis('off')
    return fig, detected_data


with st.sidebar:
    st.image("https://th.bing.com/th/id/R.7545b55b9d17b1070e2c884ffa6858fd?rik=3D80%2fEg6i9TK2A&riu=http%3a%2f%2f1.bp.blogspot.com%2f-P8KJ9GPI9ds%2fT9QrVuX-ycI%2fAAAAAAAAK3g%2fdW9fIbMoO14%2fs1600%2flogo%2bunsri.png", use_column_width=True)
    st.markdown("<h2 style='text-align: center; color: #1E293B; font-weight: 600;'>DermSight AI</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: white; padding: 16px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 20px;'>
    <b style='color: #334155;'>Fungsi Sistem:</b><br>
    <span style='color: #475569; font-size: 0.9rem;'>Analisis visual berbasis Deep Learning untuk identifikasi awal pola abnormalitas kulit.</span>
    <hr style='margin: 12px 0; border: 0; border-top: 1px solid #E2E8F0;'>
    <b style='color: #334155;'>Arsitektur:</b><br>
    <span style='color: #475569; font-size: 0.9rem;'>EfficientNetV2-S</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #FEF2F2; color: #991B1B; padding: 16px; border-radius: 8px; border-left: 4px solid #DC2626; font-size: 0.85rem;'>
    <b>DISCLAIMER MEDIS</b><br>
    Hasil analisis sistem bersifat indikatif dan tidak dapat mensubstitusi diagnosis klinis dari tenaga medis profesional.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-title">DermSight AI</div>
    <div class="hero-subtitle">Pemindaian Visual Dermatologi Cerdas</div>
</div>
""", unsafe_allow_html=True)

with st.expander("Prosedur Penggunaan Sistem"):
    st.markdown("""
    - **Langkah 1:** Unggah citra melalui panel galeri atau aktifkan tangkapan kamera langsung.
    - **Langkah 2:** Pastikan area kulit mendapatkan eksposur cahaya yang optimal dan fokus lensa yang tajam.
    - **Langkah 3:** Mulai proses analisis dan tunggu hingga indikator penyelesaian penuh.
    - **Langkah 4:** Tinjau pemetaan visual dan baca rangkuman indikasi medis yang diberikan.
    """)

tab1, tab2, tab3 = st.tabs(["Unggah Direktori Lokal", "Tangkapan Kamera", "Informasi Kondisi Kulit"])

selected_file = None

with tab1:
    st.markdown("<p style='font-weight: 500; color: #334155;'>Pilih berkas citra dari perangkat</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Format valid: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file: selected_file = uploaded_file

with tab2:
    st.markdown("<p style='font-weight: 500; color: #334155;'>Ambil citra secara langsung</p>", unsafe_allow_html=True)
    camera_file = st.camera_input("Pastikan perangkat dalam keadaan stabil")
    if camera_file: selected_file = camera_file

with tab3:
    st.markdown("<p style='font-weight: 500; color: #334155; margin-bottom: 20px;'>Klasifikasi target yang didukung oleh model analitik:</p>", unsafe_allow_html=True)
    for kondisi, info in medical_info.items():
        st.markdown(f"""
        <div class="info-card" style="border-left-color: {info['risk_text']};">
            <h3 style='margin-top: 0; color: #1E293B; font-size: 1.2rem;'>{kondisi}</h3>
            <div class="risk-badge" style="background-color: {info['risk_color']}; color: {info['risk_text']};">
                Indeks Risiko: {info['risk']}
            </div>
            <p style='color: #475569; font-size: 0.95rem;'><b>Deskripsi Klinis:</b> {info['description']}</p>
            <p style='color: #0F172A; font-size: 0.95rem; margin-bottom: 0;'><b>Saran Penanganan:</b> {info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

if selected_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with st.spinner("Menginisialisasi komputasi model..."):
        model = load_model()
        
    with col1:
        st.markdown("<h3 style='text-align:center; color:#1E293B; font-size: 1.2rem; margin-bottom: 15px;'>Citra Referensi</h3>", unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(selected_file, use_column_width=True)
        st.markdown('</div><br>', unsafe_allow_html=True)
        
        analyze_btn = st.button("Jalankan Pemindaian", use_container_width=True)

    if analyze_btn:
        selected_file.seek(0)
        with col2:
            st.markdown("<h3 style='text-align:center; color:#1E293B; font-size: 1.2rem; margin-bottom: 15px;'>Pemetaan Deteksi AI</h3>", unsafe_allow_html=True)
            result_fig, det_data = predict_image(selected_file, model)
            
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.pyplot(result_fig)
            st.markdown('</div><br>', unsafe_allow_html=True)
            
            buf = io.BytesIO(); result_fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
            st.download_button("Simpan Hasil Analisis", data=buf, file_name="Analisis_DermSight.png", mime="image/png", use_container_width=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#1E293B; font-size: 1.5rem; margin-bottom: 30px;'>Laporan Deteksi Terintegrasi</h2>", unsafe_allow_html=True)
        
        if len(det_data) > 0:
            col_table, col_info = st.columns([1, 2], gap="large")
            with col_table:
                st.markdown("<p style='font-weight: 600; color: #334155; font-size: 1.1rem;'>Detail Identifikasi:</p>", unsafe_allow_html=True)
                df_result = pd.DataFrame(det_data)
                df_result.index = df_result.index + 1
                st.dataframe(df_result, use_container_width=True)
            
            with col_info:
                st.markdown("<p style='font-weight: 600; color: #334155; font-size: 1.1rem;'>Interpretasi Medis:</p>", unsafe_allow_html=True)
                unique_diseases = set([d['Kondisi'] for d in det_data])
                for d in unique_diseases:
                    dk = d.split(" / ")[0] if " / " in d else d
                    info = medical_info.get(dk, medical_info['Normal Skin'])
                    
                    st.markdown(f"""
                    <div class="info-card" style="border-left-color: {info['risk_text']}; padding: 20px;">
                        <h4 style="margin-top: 0; color: #1E293B; font-size: 1.1rem;">{dk}</h4>
                        <div class="risk-badge" style="background-color: {info['risk_color']}; color: {info['risk_text']}; padding: 4px 10px; font-size: 0.8rem;">
                            {info['risk']}
                        </div>
                        <p style="color: #475569; font-size: 0.9rem; margin-bottom: 8px;"><b>Catatan:</b> {info['description']}</p>
                        <p style="color: #0F172A; font-size: 0.9rem; margin-bottom: 0;"><b>Tindakan:</b> {info['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else: 
            st.markdown("""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 25px; border-radius: 12px; text-align: center; color: #065F46; box-shadow: 0 4px 10px rgba(16, 185, 129, 0.05);">
                <h3 style="margin:0; color: #047857; font-size: 1.3rem;">Tidak Ditemukan Indikasi Abnormal</h3>
                <p style="font-size: 1rem; margin-top: 8px; color: #059669;">Pemindaian tidak mendeteksi pola yang mengarah pada kelainan lesi atau pigmentasi berbahaya.</p>
            </div>
            """, unsafe_allow_html=True)
            
else: 
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; background-color: #F8FAFC; border-radius: 16px; border: 2px dashed #CBD5E1; color: #64748B; margin-top: 20px;">
        <h3 style="color: #475569; font-weight: 500;">Sistem Menunggu Input</h3>
        <p>Silakan gunakan panel input di bagian atas untuk memulai prosedur analisis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Dibangun menggunakan arsitektur TensorFlow dan Streamlit</p>", unsafe_allow_html=True)
