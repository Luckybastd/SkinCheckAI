import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import io

st.set_page_config(page_title="DermSight", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #1f2937;
    }

    .header-container {
        padding: 1.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    .stDownloadButton>button {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    .stDownloadButton>button:hover {
        background-color: #e5e7eb;
        color: #111827;
        border-color: #9ca3af;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.75rem;
        margin-top: 0;
    }

    .risk-high { color: #dc2626; font-weight: 600; }
    .risk-med-high { color: #ea580c; font-weight: 600; }
    .risk-med { color: #ca8a04; font-weight: 600; }
    .risk-low { color: #16a34a; font-weight: 600; }
    .risk-safe { color: #2563eb; font-weight: 600; }

    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    hr {
        border-color: #e5e7eb;
        margin: 2rem 0;
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
        'risk': 'SANGAT TINGGI', 'risk_class': 'risk-high',
        'description': 'Jenis kanker kulit yang paling serius. Berkembang dari sel melanosit.',
        'advice': 'Segera konsultasikan dengan Dokter Spesialis Kulit untuk biopsi dan tindakan lebih lanjut.'
    },
    'Squamous Cell Carcinoma': {
        'risk': 'MENENGAH-TINGGI', 'risk_class': 'risk-med-high',
        'description': 'Kanker kulit yang berkembang di lapisan luar kulit (epidermis).',
        'advice': 'Segera temui dokter. Tingkat kesembuhan tinggi jika diobati sejak dini.'
    },
    'Basal Cell Carcinoma': {
        'risk': 'MENENGAH', 'risk_class': 'risk-med',
        'description': 'Kanker kulit yang umum. Tumbuh lambat dan jarang menyebar, namun bisa merusak jaringan sekitar.',
        'advice': 'Konsultasikan dengan dokter untuk prosedur pengangkatan jaringan.'
    },
    'Acne': {
        'risk': 'RENDAH', 'risk_class': 'risk-low',
        'description': 'Peradangan pada kelenjar minyak. Bisa berupa komedo, papula, atau pustula.',
        'advice': 'Jaga kebersihan wajah, gunakan obat jerawat, atau konsultasikan ke dokter estetika jika meradang.'
    },
    'Nevus': {
        'risk': 'AMAN', 'risk_class': 'risk-low',
        'description': 'Tahi lalat jinak. Kumpulan sel pigmen normal pada kulit yang tidak berbahaya.',
        'advice': 'Pantau secara mandiri jika ada perubahan mendadak pada bentuk, warna, atau ukurannya.'
    },
    'Normal Skin': {
        'risk': 'SANGAT AMAN', 'risk_class': 'risk-safe',
        'description': 'Kulit terlihat sehat. Tidak terdeteksi adanya kelainan atau lesi berbahaya.',
        'advice': 'Tetap jaga kesehatan kulit Anda dan gunakan tabir surya secara rutin.'
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

    progress_bar = st.progress(0, text="Memproses citra...")
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
    detected_data = [] 
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            score = confs[i]
            raw_id = ids[i]
            if isinstance(raw_id, str):
                label = raw_id; box_color = 'orange'; font_color = 'black'
            else:
                label = class_names[raw_id]
                if label == 'Melanoma': box_color = '#dc2626'; font_color = 'white'
                elif label == 'Squamous Cell Carcinoma': box_color = '#ea580c'; font_color = 'white'
                elif label == 'Basal Cell Carcinoma': box_color = '#ca8a04'; font_color = 'white'
                elif label == 'Acne':
                    box_color = '#16a34a'; font_color = 'white'
                    shrink = 0.5
                    new_w, new_h = int(w_box*shrink), int(h_box*shrink)
                    x += (w_box-new_w)//2; y += (h_box-new_h)//2
                    w_box, h_box = new_w, new_h
                else: box_color = '#2563eb'; font_color = 'white'
                
            ax.add_patch(plt.Rectangle((x, y), w_box, h_box, fill=False, color=box_color, linewidth=2))
            ax.text(x, y-8, f"{label} ({score*100:.0f}%)", color=font_color, bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='none', pad=0.3), fontsize=10, fontweight='500')
            data_entry = {'Kondisi': label, 'Probabilitas': f"{score*100:.1f}%"}
            if data_entry not in detected_data: detected_data.append(data_entry)
            
    ax.axis('off')
    return fig, detected_data

with st.sidebar:
    st.image("https://th.bing.com/th/id/R.7545b55b9d17b1070e2c884ffa6858fd?rik=3D80%2fEg6i9TK2A&riu=http%3a%2f%2f1.bp.blogspot.com%2f-P8KJ9GPI9ds%2fT9QrVuX-ycI%2fAAAAAAAAK3g%2fdW9fIbMoO14%2fs1600%2flogo%2bunsri.png", use_column_width=True)
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <h3 style='font-size: 1rem; color: #111827; font-weight: 600;'>Tentang Sistem</h3>
        <p style='font-size: 0.875rem; color: #4b5563; line-height: 1.5;'>
            Sistem deteksi dini anomali kulit berbasis deep learning menggunakan arsitektur EfficientNetV2-S.
        </p>
    </div>
    <div style='margin-top: 1.5rem; background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 1rem;'>
        <p style='font-size: 0.875rem; color: #991b1b; margin: 0; font-weight: 500;'>
            DISCLAIMER: Hasil analisis sistem ini bersifat referensi dan tidak menggantikan diagnosis medis profesional.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1 class="header-title">DermSight</h1>
    <p class="header-subtitle">Analisis Citra Dermatologi Berbasis Computer Vision</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Petunjuk Penggunaan Sistem"):
    st.markdown("""
    1. Pilih metode input citra melalui tab yang tersedia (Unggah File atau Kamera).
    2. Pastikan area kulit yang akan dianalisis terlihat jelas dan memiliki pencahayaan yang cukup.
    3. Tekan tombol 'Mulai Analisis' untuk menjalankan proses inferensi model.
    4. Hasil deteksi dan persentase probabilitas akan ditampilkan pada panel sebelah kanan.
    """)

tab1, tab2, tab3 = st.tabs(["Unggah Citra", "Kamera", "Informasi Kondisi Medis"])

selected_file = None

with tab1:
    uploaded_file = st.file_uploader("Format yang didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file: selected_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Ambil citra kulit")
    if camera_file: selected_file = camera_file

with tab3:
    st.markdown("<div style='margin-bottom: 1rem;'>Daftar kondisi kulit yang dapat diidentifikasi oleh sistem:</div>", unsafe_allow_html=True)
    for kondisi, info in medical_info.items():
        st.markdown(f"""
        <div class="card">
            <h3 class="card-title">{kondisi}</h3>
            <div style="margin-bottom: 0.5rem; font-size: 0.875rem;">
                Tingkat Risiko: <span class="{info['risk_class']}">{info['risk']}</span>
            </div>
            <p style='color: #4b5563; font-size: 0.95rem; margin-bottom: 0.5rem;'><b>Deskripsi:</b> {info['description']}</p>
            <p style='color: #111827; font-size: 0.95rem; margin: 0;'><b>Saran Medis:</b> {info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

if selected_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with st.spinner("Memuat model..."):
        model = load_model()
        
    with col1:
        st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;'>Citra Input</h3>", unsafe_allow_html=True)
        st.markdown('<div style="border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.image(selected_file, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze_btn = st.button("Mulai Analisis", use_container_width=True)

    if analyze_btn:
        selected_file.seek(0)
        with col2:
            st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;'>Hasil Deteksi</h3>", unsafe_allow_html=True)
            result_fig, det_data = predict_image(selected_file, model)
            
            st.markdown('<div style="border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.pyplot(result_fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            buf = io.BytesIO(); result_fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
            st.download_button("Unduh Hasil Deteksi", data=buf, file_name="DermSight_Result.png", mime="image/png", use_container_width=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.25rem; font-weight: 600; margin-bottom: 1.5rem;'>Rangkuman Analisis</h3>", unsafe_allow_html=True)
        
        if len(det_data) > 0:
            col_table, col_info = st.columns([1, 2], gap="large")
            with col_table:
                st.markdown("<div style='font-weight: 500; margin-bottom: 0.5rem;'>Detail Temuan:</div>", unsafe_allow_html=True)
                df_result = pd.DataFrame(det_data)
                df_result.index = df_result.index + 1
                st.dataframe(df_result, use_container_width=True)
            
            with col_info:
                st.markdown("<div style='font-weight: 500; margin-bottom: 0.5rem;'>Keterangan Tambahan:</div>", unsafe_allow_html=True)
                unique_diseases = set([d['Kondisi'] for d in det_data])
                for d in unique_diseases:
                    dk = d.split(" / ")[0] if " / " in d else d
                    info = medical_info.get(dk, medical_info['Normal Skin'])
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4 style="margin-top: 0; font-size: 1.1rem; color: #111827;">{dk}</h4>
                        <div style="margin-bottom: 0.5rem; font-size: 0.875rem;">
                            Tingkat Risiko: <span class="{info['risk_class']}">{info['risk']}</span>
                        </div>
                        <p style="color: #4b5563; font-size: 0.95rem; margin-bottom: 0.5rem;"><b>Keterangan:</b> {info['description']}</p>
                        <p style="color: #111827; font-size: 0.95rem; margin: 0;"><b>Tindakan:</b> {info['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else: 
            st.info("Sistem tidak mendeteksi adanya indikasi anomali atau lesi pada area yang dianalisis.")
            
else: 
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; background-color: #f9fafb; border: 1px dashed #d1d5db; border-radius: 8px; color: #6b7280; margin-top: 1.5rem;">
        <p style="font-size: 1rem; margin: 0;">Menunggu input citra. Silakan unggah atau ambil foto untuk memulai analisis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 3rem;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.875rem;'>DermSight v1.0 | Menggunakan antarmuka Streamlit</p>", unsafe_allow_html=True)
