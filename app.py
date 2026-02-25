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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #0B2447, #19376D, #576CBC);
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(25, 55, 109, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        background: linear-gradient(to right, #A5D7E8, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .stButton>button {
        background: linear-gradient(90deg, #19376D 0%, #576CBC 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(87, 108, 188, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(87, 108, 188, 0.6);
        transform: translateY(-2px);
        color: white;
        border-color: transparent;
    }

    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
        border-left: 6px solid #576CBC;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    
    .risk-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(0,0,0,0), rgba(87, 108, 188, 0.3), rgba(0,0,0,0));
        margin: 30px 0;
    }

    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 15px 0 25px 0;
        font-size: 1.05em;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #CBD5E1;
    }
    
    .custom-table thead tr {
        background-color: #1E293B;
        color: #FFFFFF;
        text-align: left;
        font-weight: 600;
    }
    
    .custom-table th {
        padding: 16px 20px;
        border-bottom: 2px solid #0F172A;
    }
    
    .custom-table td {
        padding: 16px 20px;
    }
    
    .custom-table tbody tr {
        border-bottom: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #0F172A;
        font-weight: 500;
    }
    
    .custom-table tbody tr:nth-of-type(even) {
        background-color: #F8FAFC;
    }
    
    .custom-table tbody tr:last-of-type {
        border-bottom: none;
    }
    
    .custom-table tbody tr:hover {
        background-color: #E2E8F0;
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
        'risk': 'VERY HIGH',
        'risk_color': '#FEE2E2', 'risk_text': '#DC2626',
        'description': 'The most deadly type of skin cancer if left untreated. Develops from pigment-producing melanocytes.',
        'advice': 'Consult a Dermatologist immediately for a biopsy and further medical action. Do not delay.'
    },
    'Squamous Cell Carcinoma': {
        'risk': 'MODERATE-HIGH',
        'risk_color': '#FEF3C7', 'risk_text': '#D97706',
        'description': 'Skin cancer that develops in the outer epidermal layer. Usually appears in sun-exposed areas.',
        'advice': 'See a doctor immediately. The cure rate is very high if detected and treated early.'
    },
    'Basal Cell Carcinoma': {
        'risk': 'MODERATE',
        'risk_color': '#FEF9C3', 'risk_text': '#CA8A04',
        'description': 'The most common skin cancer. Grows slowly and rarely spreads to other organs, but can damage surrounding tissue.',
        'advice': 'Consult a doctor for a tissue removal procedure so it does not enlarge and damage aesthetics.'
    },
    'Acne': {
        'risk': 'LOW',
        'risk_color': '#D1FAE5', 'risk_text': '#059669',
        'description': 'Common acne. Occurs due to inflammation of the oil glands. Can be blackheads, papules, or pustules.',
        'advice': 'Maintain facial hygiene, reduce oily food, use acne medication, or visit an aesthetic doctor if severely inflamed.'
    },
    'Nevus': {
        'risk': 'SAFE',
        'risk_color': '#E0F2FE', 'risk_text': '#0284C7',
        'description': 'Benign mole. A collection of normal pigment cells on the skin that is completely harmless.',
        'advice': 'No medical action needed. However, monitor if there are sudden changes in shape, color, or size.'
    },
    'Normal Skin': {
        'risk': 'VERY SAFE',
        'risk_color': '#DBEAFE', 'risk_text': '#1D4ED8',
        'description': 'The skin looks healthy. No dangerous lesions, abnormalities, or infections detected.',
        'advice': 'Keep maintaining your skin health and use sunscreen regularly.'
    }
}

risk_rank = {
    'VERY HIGH': 5,
    'MODERATE-HIGH': 4,
    'MODERATE': 3,
    'LOW': 2,
    'SAFE': 1,
    'VERY SAFE': 0
}

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"File {WEIGHTS_PATH} not found.")
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
            st.error(f"Failed to load model: {e_final}")
            st.stop()

def process_results(preds_batch, coords, boxes, confidences, class_ids):
    for i, preds in enumerate(preds_batch):
        x, y = coords[i]
        p_bcc, p_mel, p_scc = preds[1], preds[2], preds[5]
        
        if p_mel > 0.30 or p_scc > 0.30 or p_bcc > 0.30:
            suspects = []
            if p_mel > 0.20: suspects.append((p_mel, "Melanoma"))
            if p_scc > 0.20: suspects.append((p_scc, "Squamous Cell Carcinoma"))
            if p_bcc > 0.20: suspects.append((p_bcc, "Basal Cell Carcinoma"))
            suspects.sort(key=lambda item: item[0], reverse=True)
            
            if suspects and suspects[0][0] > 0.30:
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
                class_ids.append(label)

def predict_image(image_file, model):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = original_img.shape
    if h < IMG_SIZE or w < IMG_SIZE:
        scale = max(IMG_SIZE/h, IMG_SIZE/w)
        original_img = cv2.resize(original_img, (int(w*scale)+1, int(h*scale)+1))
        h, w, _ = original_img.shape

    progress_bar = st.progress(0, text="AI is scanning your skin texture...")
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
            label = ids[i]
            
            if label == 'Melanoma': box_color = '#DC2626'; font_color = 'white'
            elif label == 'Squamous Cell Carcinoma': box_color = '#D97706'; font_color = 'white'
            elif label == 'Basal Cell Carcinoma': box_color = '#CA8A04'; font_color = 'white'
            elif label == 'Acne':
                box_color = '#059669'; font_color = 'white'
                shrink = 0.5
                new_w, new_h = int(w_box*shrink), int(h_box*shrink)
                x += (w_box-new_w)//2; y += (h_box-new_h)//2
                w_box, h_box = new_w, new_h
            else: box_color = '#0284C7'; font_color = 'white'
            
            ax.add_patch(plt.Rectangle((x, y), w_box, h_box, fill=False, color=box_color, linewidth=4))
            ax.text(x, y-10, f"{label} ({score*100:.0f}%)", color=font_color, bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'), fontsize=11, fontweight='bold')
            
            if label not in highest_confidence_per_class or score > highest_confidence_per_class[label]:
                highest_confidence_per_class[label] = score
                
    detected_data = []
    for label, score in highest_confidence_per_class.items():
        detected_data.append({'Condition': label, 'AI Accuracy': f"{score*100:.1f}%"})
        
    detected_data = sorted(detected_data, key=lambda val: float(val['AI Accuracy'].strip('%')), reverse=True)
            
    ax.axis('off')
    return fig, detected_data

with st.sidebar:
    st.image("https://i0.wp.com/tambahpinter.com/wp-content/uploads/2020/05/logo-unsri-840x596.png", use_column_width=True)
    st.markdown("<h2 style='text-align: center; color: #19376D;'>DermSight</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;'>
    <b>Purpose:</b><br>
    Helping the public detect early skin problems through the lens of Artificial Intelligence.
    <hr style='margin: 10px 0;'>
    <b>Architecture:</b><br>
    EfficientNetV2-S
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #FEF2F2; color: #991B1B; padding: 15px; border-radius: 10px; border-left: 5px solid #DC2626; font-size: 0.9rem;'>
    <b>MEDICAL DISCLAIMER</b><br>
    This system is not a substitute for a doctor. It serves only as a reference and initial detection tool. If in doubt, always visit the nearest health facility.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-title">DermSight</div>
    <div class="hero-subtitle">Empowering Early Detection of Dermatological Conditions through Computer Vision and Real Time Camera Diagnostics.</div>
</div>
""", unsafe_allow_html=True)

with st.expander("Application Usage Guide (Must Read)"):
    st.markdown("""
    - **Step 1:** Select the upload tab from the Gallery or directly use your Camera.
    - **Step 2:** Use bright lighting, ensure the photo is focused and clearly shows the skin area.
    - **Step 3:** Click the Start AI Analysis button and wait for the engine to work.
    - **Step 4:** Read the results box on the right to see medical advice.
    """)

tab1, tab2, tab3 = st.tabs(["Upload Gallery", "Use Camera", "Skin Encyclopedia"])

selected_file = None

with tab1:
    st.markdown("### Choose a photo from your device")
    uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file: selected_file = uploaded_file

with tab2:
    st.markdown("### Take a picture of your skin directly")
    camera_file = st.camera_input("Ensure bright light and focused image")
    if camera_file: selected_file = camera_file

with tab3:
    st.markdown("### Understand Various Skin Conditions")
    st.markdown("Here is a list of conditions trained and recognizable by our AI model:")
    for kondisi, info in medical_info.items():
        st.markdown(f"""
        <div class="info-card" style="border-left-color: {info['risk_text']};">
            <h3 style='margin-top: 0; color: #19376D;'>{kondisi}</h3>
            <div class="risk-badge" style="background-color: {info['risk_color']}; color: {info['risk_text']};">
                Risk: {info['risk']}
            </div>
            <p style='color: #4B5563; font-size: 1.05rem;'><b>Description:</b> {info['description']}</p>
            <p style='color: #374151; font-style: italic;'><b>Medical Advice:</b> {info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

if selected_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with st.spinner("Initializing AI Model..."):
        model = load_model()
        
    with col1:
        st.markdown("<h3 style='text-align:center; color:#19376D;'>Input Photo</h3>", unsafe_allow_html=True)
        st.markdown('<div style="border-radius:15px; overflow:hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        st.image(selected_file, use_column_width=True)
        st.markdown('</div><br>', unsafe_allow_html=True)
        
        analyze_btn = st.button("Start AI Analysis", use_container_width=True)

    if analyze_btn:
        selected_file.seek(0)
        with col2:
            st.markdown("<h3 style='text-align:center; color:#19376D;'>Scan Results</h3>", unsafe_allow_html=True)
            result_fig, det_data = predict_image(selected_file, model)
            
            st.markdown('<div style="border-radius:15px; overflow:hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 2px solid #576CBC;">', unsafe_allow_html=True)
            st.pyplot(result_fig)
            st.markdown('</div><br>', unsafe_allow_html=True)
            
            buf = io.BytesIO(); result_fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
            st.download_button("Save Results to Gallery", data=buf, file_name="DermSight_Result.png", mime="image/png", use_container_width=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#19376D; margin-bottom: 30px;'>Your Medical Summary</h2>", unsafe_allow_html=True)
        
        if len(det_data) > 0:
            col_table, col_info = st.columns([1, 2], gap="large")
            with col_table:
                st.markdown("<h4 style='color: #1E293B; margin-bottom: 0;'>AI Findings Detail:</h4>", unsafe_allow_html=True)
                
                table_html = "<table class='custom-table'><thead><tr><th style='width: 10%;'>#</th><th>Condition</th><th>AI Accuracy</th></tr></thead><tbody>"
                for idx, row in enumerate(det_data):
                    table_html += f"<tr><td><b>{idx+1}</b></td><td>{row['Condition']}</td><td>{row['AI Accuracy']}</td></tr>"
                table_html += "</tbody></table>"
                
                st.markdown(table_html, unsafe_allow_html=True)
            
            with col_info:
                st.markdown("<h4 style='color: #1E293B; margin-bottom: 15px;'>Explanation & Action:</h4>", unsafe_allow_html=True)
                
                unique_diseases = list(set([d['Condition'] for d in det_data]))
                unique_diseases.sort(key=lambda x: risk_rank[medical_info.get(x, medical_info['Normal Skin'])['risk']], reverse=True)
                
                for dk in unique_diseases:
                    info = medical_info.get(dk, medical_info['Normal Skin'])
                    
                    st.markdown(f"""
                    <div class="info-card" style="border-left-color: {info['risk_text']}; margin-bottom: 25px;">
                        <h3 style="margin-top: 0; margin-bottom: 15px; color: #1E293B;">{dk}</h3>
                        <div class="risk-badge" style="background-color: {info['risk_color']}; color: {info['risk_text']}; padding: 6px 14px; font-size: 0.8rem; margin-bottom: 20px;">
                            {info['risk']}
                        </div>
                        <p style="color: #475569; font-size: 0.95rem; margin-bottom: 15px;"><b>Explanation:</b> {info['description']}</p>
                        <p style="color: #0F172A; font-size: 0.95rem; margin-bottom: 0;"><b>Action:</b> <em>{info['advice']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        else: 
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10B981, #059669); padding: 30px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);">
                <h2 style="margin:0; color: white;">Good News</h2>
                <p style="font-size: 1.1rem; margin-top: 10px;">The AI found no indications of dangerous abnormalities such as cancer in the scanned area. Your skin looks healthy.</p>
            </div>
            """, unsafe_allow_html=True)
            
else: 
    st.markdown("""
    <div style="text-align:center; padding: 50px; background-color: #F8F9FA; border-radius: 15px; border: 2px dashed #CBD5E1; color: #64748B; margin-top: 20px;">
        <h2 style="color: #CBD5E1; margin-bottom: 10px;">Image Preview Area</h2>
        <h3 style="color: #64748B;">Waiting for Image</h3>
        <p style="color: #94A3B8;">Please upload a photo or take a picture via the tabs above to start the analysis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.9rem;'>Designed & Developed by jnn</p>", unsafe_allow_html=True)
