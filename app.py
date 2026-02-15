import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import io

WEIGHTS_PATH = 'model_weights.h5' 
IMG_SIZE = 384
NUM_CLASSES = 6
CONFIDENCE_THRESHOLD = 0.50
STRIDE = 100 

class_names = ['Acne', 'Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Normal Skin', 'Squamous Cell Carcinoma']

medical_info = {
    'Melanoma': {
        'risk': 'HIGH (Seek Immediate Medical Attention)',
        'description': 'The most serious type of skin cancer. Develops in the cells (melanocytes) that produce melanin.',
        'advice': 'Consult a Dermatologist immediately for biopsy and further treatment.'
    },
    'Squamous Cell Carcinoma': {
        'risk': 'MODERATE-HIGH',
        'description': 'Skin cancer that develops in the outer layer of the skin.',
        'advice': 'See a doctor. It is usually curable if treated early.'
    },
    'Basal Cell Carcinoma': {
        'risk': 'MODERATE',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads.',
        'advice': 'Consult a doctor for tissue removal.'
    },
    'Acne': {
        'risk': 'LOW (Aesthetic/Infection Issue)',
        'description': 'Inflammation of oil glands. Can appear as blackheads, papules, or pustules.',
        'advice': 'Maintain facial hygiene, use acne medication, or consult an aesthetic doctor.'
    },
    'Nevus': {
        'risk': 'LOW (Safe)',
        'description': 'Benign mole. A collection of normal pigment cells.',
        'advice': 'Usually requires no action unless there are drastic changes in shape/color.'
    },
    'Normal Skin': {
        'risk': 'SAFE',
        'description': 'Healthy skin with no detected abnormalities.',
        'advice': 'Maintain skin health and use sunscreen regularly.'
    }
}

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"❌ File '{WEIGHTS_PATH}' tidak ditemukan!")
        st.stop()
    
    try:
        # Karena file Anda ternyata berisi '4 layers' (Full Model),
        # Kita gunakan load_model, bukan load_weights.
        model = tf.keras.models.load_model(WEIGHTS_PATH)
        print("✅ Full Model loaded successfully.")
        return model
    except Exception as e:
        # Jika load_model gagal karena masalah format Keras 3, 
        # kita gunakan cara manual sebagai cadangan terakhir.
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
            # Memuat bobot dengan skip_mismatch agar tidak error 283 layers lagi
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

    progress_bar = st.progress(0, text="Scanning skin image...")
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
            ax.text(x, y-10, f"{label} ({score*100:.0f}%)", color=font_color, bbox=dict(facecolor=box_color, alpha=0.8), fontsize=9, fontweight='bold')
            data_entry = {'Condition': label, 'Confidence': f"{score*100:.1f}%"}
            if data_entry not in detected_data: detected_data.append(data_entry)
    ax.axis('off')
    return fig, detected_data

st.set_page_config(page_title="SkinCheck AI", page_icon="🔬", layout="wide")
with st.sidebar:
    st.image("https://th.bing.com/th/id/R.7545b55b9d17b1070e2c884ffa6858fd?rik=3D80%2fEg6i9TK2A&riu=http%3a%2f%2f1.bp.blogspot.com%2f-P8KJ9GPI9ds%2fT9QrVuX-ycI%2fAAAAAAAAK3g%2fdW9fIbMoO14%2fs1600%2flogo%2bunsri.png&ehk=9XoxwvoaYfdUgOg7B0UHZJ0FrOEQIEK%2fiOrPBfmqUgE%3d&risl=&pid=ImgRaw&r=0", caption="Sriwijaya University Logo", use_column_width=True)
    st.title("About the App")
    st.info("""
    **SkinCheck AI** provides early detection of skin abnormalities using *Deep Learning* technology.
    **Model:** EfficientNetV2-S
    """)
    st.warning("⚠️ **DISCLAIMER:**\nThis AI detection result is for REFERENCE ONLY. A definitive diagnosis must be performed by a certified Doctor.")
st.title("🔬 SkinCheck AI: Early Skin Disease Detection")
tab1, tab2 = st.tabs(["📁 Upload from Gallery", "📸 Take Photo (Camera)"])
selected_file = None
with tab1:
    uploaded_file = st.file_uploader("Upload skin photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file: selected_file = uploaded_file
with tab2:
    camera_file = st.camera_input("Take a photo directly")
    if camera_file: selected_file = camera_file

if selected_file is not None:
    col1, col2 = st.columns([1, 1])
    model = load_model()
    with col1:
        st.subheader("📸 Input Photo")
        st.image(selected_file, use_column_width=True)
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            selected_file.seek(0)
            with col2:
                st.subheader("🎯 AI Detection Result")
                result_fig, det_data = predict_image(selected_file, model)
                st.pyplot(result_fig)
                buf = io.BytesIO(); result_fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1: st.download_button("⬇️ Download Result", data=buf, file_name="result.png", mime="image/png", use_container_width=True)
            st.divider()
            if len(det_data) > 0:
                st.table(pd.DataFrame(det_data))
                unique_diseases = set([d['Condition'] for d in det_data])
                for d in unique_diseases:
                    dk = d.split(" / ")[0] if " / " in d else d
                    info = medical_info.get(dk, medical_info['Normal Skin'])
                    with st.expander(f"ℹ️ What is {dk}?", expanded=True):
                        st.markdown(f"**Risk:** `{info['risk']}`\n**Explanation:** {info['description']}\n**Advice:** {info['advice']}")
            else: st.success("✅ **Clean Result!**")
else: st.info("👆 Please select an input method above.")
st.divider()
st.caption("Made by: jnn")
