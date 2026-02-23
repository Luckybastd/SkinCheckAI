import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import io

# 1. KONFIGURASI HALAMAN WAJIB DI ATAS
st.set_page_config(page_title="DermSight AI", page_icon="🔬", layout="wide")

# 2. CUSTOM CSS UNTUK TAMPILAN LEBIH MENARIK
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #6B7280;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: scale(1.02);
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
        'risk': 'SANGAT TINGGI 🔴 (Butuh Penanganan Medis Segera)',
        'description': 'Jenis kanker kulit yang paling serius dan mematikan jika dibiarkan. Berkembang dari sel yang memproduksi pigmen melanin.',
        'advice': 'Segera konsultasikan dengan Dokter Spesialis Kulit (Dermatolog) untuk biopsi dan tindakan lebih lanjut. Jangan ditunda!'
    },
    'Squamous Cell Carcinoma': {
        'risk': 'MENENGAH-TINGGI 🟠',
        'description': 'Kanker kulit yang berkembang di lapisan luar kulit (epidermis). Biasanya muncul di area yang sering terkena sinar matahari.',
        'advice': 'Segera temui dokter. Tingkat kesembuhan sangat tinggi jika terdeteksi dan diobati sejak dini.'
    },
    'Basal Cell Carcinoma': {
        'risk': 'MENENGAH 🟡',
        'description': 'Jenis kanker kulit yang paling umum. Tumbuh sangat lambat dan jarang menyebar ke bagian tubuh lain, namun bisa merusak jaringan sekitar.',
        'advice': 'Konsultasikan dengan dokter untuk prosedur pengangkatan jaringan agar tidak membesar.'
    },
    'Acne': {
        'risk': 'RENDAH 🟢 (Masalah Estetika/Infeksi Ringan)',
        'description': 'Jerawat. Terjadi akibat peradangan pada kelenjar minyak. Bisa berupa komedo, papula, atau pustula bernanah.',
        'advice': 'Jaga kebersihan wajah, gunakan obat jerawat yang dijual bebas, atau konsultasikan dengan dokter estetika jika meradang hebat.'
    },
    'Nevus': {
        'risk': 'RENDAH 🟢 (Aman)',
        'description': 'Tahi lalat jinak. Merupakan kumpulan sel pigmen normal pada kulit yang tidak berbahaya.',
        'advice': 'Umumnya tidak perlu tindakan medis. Namun, pantau jika ada perubahan drastis pada bentuk, warna, atau ukuran.'
    },
    'Normal Skin': {
        'risk': 'AMAN 🔵',
        'description': 'Kulit terlihat sehat dan tidak terdeteksi adanya kelainan atau lesi berbahaya.',
        'advice': 'Tetap jaga kesehatan kulit Anda! Gunakan tabir surya (sunscreen) secara rutin saat beraktivitas di luar ruangan.'
    }
}

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"⚠️ File '{WEIGHTS_PATH}' tidak ditemukan! Pastikan file model ada di direktori yang benar.")
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
            st.error(f"❌ Gagal memuat model: {e_final}")
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

    progress_bar = st.progress(0, text="🔍 Sedang memindai gambar kulit Anda...")
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
                if label == 'Melanoma': box_color = 'red'; font_color = 'white'
                elif label == 'Squamous Cell Carcinoma': box_color = 'magenta'; font_color = 'white'
                elif label == 'Basal Cell Carcinoma': box_color = 'blue'; font_color = 'white'
                elif label == 'Acne':
                    box_color = 'cyan'; font_color = 'black'
                    shrink = 0.5
                    new_w, new_h = int(w_box*shrink), int(h_box*shrink)
                    x += (w_box-new_w)//2; y += (h_box-new_h)//2
                    w_box, h_box = new_w, new_h
                else: box_color = '#00FF00'; font_color = 'black'
                
            ax.add_patch(plt.Rectangle((x, y), w_box, h_box, fill=False, color=box_color, linewidth=3))
            ax.text(x, y-10, f"{label} ({score*100:.0f}%)", color=font_color, bbox=dict(facecolor=box_color, alpha=0.8), fontsize=10, fontweight='bold')
            data_entry = {'Kondisi Terdeteksi': label, 'Tingkat Keyakinan (Confidence)': f"{score*100:.1f}%"}
            if data_entry not in detected_data: detected_data.append(data_entry)
            
    ax.axis('off')
    return fig, detected_data


# --- STRUKTUR UI WEB ---

# Sidebar
with st.sidebar:
    st.image("https://th.bing.com/th/id/R.7545b55b9d17b1070e2c884ffa6858fd?rik=3D80%2fEg6i9TK2A&riu=http%3a%2f%2f1.bp.blogspot.com%2f-P8KJ9GPI9ds%2fT9QrVuX-ycI%2fAAAAAAAAK3g%2fdW9fIbMoO14%2fs1600%2flogo%2bunsri.png&ehk=9XoxwvoaYfdUgOg7B0UHZJ0FrOEQIEK%2fiOrPBfmqUgE%3d&risl=&pid=ImgRaw&r=0", use_column_width=True)
    st.title("Tentang Aplikasi")
    st.markdown("""
    **DermSight AI** dikembangkan untuk membantu deteksi dini masalah kulit menggunakan teknologi *Deep Learning* (Kecerdasan Buatan).
    
    ⚙️ **Mesin AI:** EfficientNetV2-S
    """)
    st.warning("⚠️ **PERHATIAN:**\nHasil deteksi AI ini HANYA SEBAGAI REFERENSI AWAL. Tidak dapat menggantikan diagnosis medis profesional. Selalu konsultasikan dengan Dokter Spesialis Kulit.")

# Main Page Header
st.markdown('<p class="main-header">🔬 DermSight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deteksi Dini Kondisi Kulit Anda dengan Kecerdasan Buatan secara Real-Time</p>', unsafe_allow_html=True)

# Petunjuk Penggunaan
with st.expander("📖 **Cara Menggunakan Aplikasi (Klik untuk membaca)**"):
    st.markdown("""
    1. **Pilih Metode:** Gunakan tab di bawah untuk memilih apakah Anda ingin mengunggah foto dari galeri atau memotret langsung dari kamera.
    2. **Pastikan Pencahayaan Bagus:** Untuk hasil terbaik, pastikan foto kulit terlihat jelas, fokus, dan tidak gelap.
    3. **Mulai Analisis:** Klik tombol **"Mulai Analisis AI"** dan tunggu beberapa detik.
    4. **Baca Hasil:** AI akan menyorot area yang mencurigakan dan memberikan penjelasan medis di bawah foto.
    """)

# Tabs (Ditambah tab Edukasi agar informatif)
tab1, tab2, tab3 = st.tabs(["📁 Unggah dari Galeri", "📷 Ambil Foto (Kamera)", "📚 Kamus Penyakit Kulit"])

selected_file = None

with tab1:
    uploaded_file = st.file_uploader("Pilih foto kulit Anda (Format: JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file: selected_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Ambil foto kulit secara langsung")
    if camera_file: selected_file = camera_file

with tab3:
    st.subheader("Pahami Berbagai Kondisi Kulit")
    st.markdown("Berikut adalah daftar kondisi yang dapat dideteksi oleh sistem kami:")
    for kondisi, info in medical_info.items():
        if kondisi != 'Normal Skin':
            with st.expander(f"🔎 {kondisi}"):
                st.markdown(f"**Tingkat Risiko:** {info['risk']}\n\n**Deskripsi:** {info['description']}\n\n**Saran Medis:** {info['advice']}")

# Proses Prediksi
if selected_file is not None:
    st.divider()
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with st.spinner("Mempersiapkan mesin AI..."):
        model = load_model()
        
    with col1:
        st.subheader("📷 Foto Input Anda")
        st.image(selected_file, use_column_width=True, caption="Siap untuk dianalisis")
        analyze_btn = st.button("🚀 Mulai Analisis AI", use_container_width=True)

    if analyze_btn:
        selected_file.seek(0)
        with col2:
            st.subheader("🩺 Hasil Deteksi AI")
            result_fig, det_data = predict_image(selected_file, model)
            st.pyplot(result_fig)
            
            # Tombol Download
            buf = io.BytesIO(); result_fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
            st.download_button("💾 Unduh Gambar Hasil Deteksi", data=buf, file_name="DermSight_Result.png", mime="image/png", use_container_width=True)
            
        st.divider()
        st.subheader("📊 Rangkuman Analisis")
        
        if len(det_data) > 0:
            st.warning("⚠️ **Terdapat kondisi yang perlu diperhatikan pada kulit Anda. Silakan baca detailnya di bawah ini.**")
            
            # Tampilkan Tabel
            df_result = pd.DataFrame(det_data)
            df_result.index = df_result.index + 1
            st.table(df_result)
            
            st.markdown("### 💡 Informasi Medis Terkait:")
            unique_diseases = set([d['Kondisi Terdeteksi'] for d in det_data])
            for d in unique_diseases:
                dk = d.split(" / ")[0] if " / " in d else d
                info = medical_info.get(dk, medical_info['Normal Skin'])
                
                # Desain kotak informasi penyakit
                st.markdown(f"""
                <div style="background-color: #F8FAFC; padding: 20px; border-radius: 10px; border-left: 6px solid #F59E0B; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h4 style="margin-top: 0; color: #1E293B;">{dk}</h4>
                    <p><strong>Tingkat Risiko:</strong> {info['risk']}</p>
                    <p><strong>Penjelasan:</strong> {info['description']}</p>
                    <p><strong>Saran:</strong> <em>{info['advice']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
        else: 
            st.success("🎉 **Kabar Baik!** AI tidak mendeteksi adanya kelainan berbahaya (Acne/Kanker) pada gambar yang dipindai. Tetap jaga kesehatan kulit Anda!")
            
else: 
    st.info("👆 Silakan pilih foto dari galeri atau gunakan kamera untuk memulai.")

st.divider()
st.caption("Dibuat dengan ❤️ oleh jnn | Didukung oleh TensorFlow & Streamlit")
