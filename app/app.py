import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pickle
import os
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# Page Setup & Aesthetic Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Clinical Exanthem Triage Engine",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Clinical Exanthem Triage Pipeline")
st.markdown("""
This system acts as an intelligent diagnostic gatekeeper. It screens incoming clinical lesion images 
for anomalies using an **Out-of-Distribution (OOD) Detector** before passing known sample profiles to 
the downstream **Random Forest Binary Classifier**.
""")
st.write("---")

# -----------------------------------------
# Core Image Processing & Feature Extraction
# -----------------------------------------
class ExanthemFeatureExtractor:
    """Extracts identical 8-dimensional tabular signatures from raw lesion images."""
    def __init__(self):
        self.feature_names = ['lesion_count', 'avg_area', 'std_area', 'avg_circularity',
                              'sparsity_score', 'confluence_ratio', 'avg_hue', 'avg_saturation']

    def apply_gray_world_white_balance(self, img):
        b, g, r = cv2.split(img.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        scale_b = avg_all / avg_b if avg_b > 0 else 1.0
        scale_g = avg_all / avg_g if avg_g > 0 else 1.0
        scale_r = avg_all / avg_r if avg_r > 0 else 1.0
        return cv2.merge((np.clip(b * scale_b, 0, 255),
                          np.clip(g * scale_g, 0, 255),
                          np.clip(r * scale_r, 0, 255))).astype(np.uint8)

    def extract_tabular_features(self, img):
        img = cv2.resize(img, (512, 512))
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids, areas, circularities = [], [], []
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < (512 * 512 * 0.1):
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
                    areas.append(area)
                    circularities.append(circularity)
                    valid_contours.append(cnt)

        wb_img = self.apply_gray_world_white_balance(img)
        hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
        valid_mask = np.zeros((512, 512), dtype=np.uint8)
        if valid_contours:
            cv2.drawContours(valid_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(hsv_img, mask=valid_mask)
            avg_hue, avg_saturation = mean_color[0], mean_color[1]
        else:
            avg_hue, avg_saturation = 0, 0

        std_area = np.std(areas) if len(areas) > 1 else 0
        avg_circularity = np.mean(circularities) if circularities else 0

        if len(centroids) > 1:
            dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
            np.fill_diagonal(dist_matrix, np.inf)
            sparsity_score = np.mean(np.min(dist_matrix, axis=1))
        else:
            sparsity_score = 0

        confluence_ratio = sum(areas) / (512 * 512)
        return [len(centroids), np.mean(areas) if areas else 0, std_area,
                avg_circularity, sparsity_score, confluence_ratio, avg_hue, avg_saturation]

# -----------------------------------------
# Dynamic Parameter & Demo Generation 
# -----------------------------------------
def generate_demo_artifacts():
    """Generates synthetic parameters for instant testing if local pkl files don't exist."""
    np.random.seed(42)
    mock_train_data = np.random.randn(100, 8)
    
    # Calculate empirical In-Distribution properties
    mean_vec = np.mean(mock_train_data, axis=0)
    cov_mat = np.cov(mock_train_data, rowvar=False)
    inv_cov_mat = np.linalg.inv(cov_mat)
    
    scaler = StandardScaler()
    scaler.fit(mock_train_data)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    mock_labels = np.random.randint(0, 2, size=100)
    rf.fit(mock_train_data, mock_labels)
    
    ood_meta = {'mean': mean_vec, 'inv_covariance': inv_cov_mat}
    return scaler, rf, ood_meta

# -----------------------------------------
# Sidebar Artifact Controls
# -----------------------------------------
st.sidebar.header("📁 Pipeline Configuration")
mode = st.sidebar.radio("Resource Mode", ["Use Synthetic Demo Parameters", "Upload Production Models"])

if mode == "Use Synthetic Demo Parameters":
    scaler, classifier, ood_parameters = generate_demo_artifacts()
    st.sidebar.success("Loaded synthetic pipeline configurations.")
else:
    scaler_file = st.sidebar.file_uploader("Upload StandardScaler (final_scaler.pkl)", type=["pkl"])
    model_file = st.sidebar.file_uploader("Upload Model (production_random_forest.pkl)", type=["pkl"])
    
    if scaler_file and model_file:
        scaler = pickle.load(scaler_file)
        classifier = pickle.load(model_file)
        
        # Calculate OOD parameters dynamically based on the scaler's training state
        # Utilizing the scaler's empirical means and variances as a proxy dataset profile
        mean_vec = scaler.mean_
        scale_vec = scaler.scale_
        # Generate a standard identity matrix structure representing the scaled training distribution space
        inv_cov_mat = np.linalg.inv(np.diag(scale_vec ** 2))
        ood_parameters = {'mean': mean_vec, 'inv_covariance': inv_cov_mat}
        st.sidebar.success("All production models parsed successfully.")
    else:
        st.sidebar.warning("Awaiting production pickle configuration files...")
        st.stop()

# Interactive Triage Sensitivity
st.sidebar.header("⚙️ OOD Sensitivity Tuning")
st.sidebar.markdown("""
Adjust the anomaly threshold. Higher settings are more lenient, while lower parameters route more cases to human experts.
""")
ood_threshold = st.sidebar.slider("Mahalanobis Distance Threshold ($D_M$)", min_value=1.0, max_value=15.0, value=4.5, step=0.1)

# -----------------------------------------
# Main Application Execution Framework
# -----------------------------------------
uploaded_img = st.file_uploader("📥 Upload Patient Lesion Image Asset", type=["png", "jpg", "jpeg"])

if uploaded_img is not None:
    # Read and decode the image array
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    raw_bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    display_rgb_image = cv2.cvtColor(raw_bgr_image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(display_rgb_image, caption="Patient Input Scan Asset", use_container_width=True)
        
    with col2:
        st.subheader("🛠️ Step 1: Feature Extraction")
        with st.spinner("Analyzing physiological characteristics..."):
            extractor = ExanthemFeatureExtractor()
            raw_features = extractor.extract_tabular_features(raw_bgr_image)
            feature_vector = np.array(raw_features).reshape(1, -1)
            
            # Display feature monitoring grid
            df_feats = pd.DataFrame(feature_vector, columns=extractor.feature_names)
            st.dataframe(df_feats.T.rename(columns={0: "Extracted Scalar Value"}))

        # --- Pipeline Stage 1: Out of Distribution Detection ---
        st.subheader("🔍 Step 2: Mahalanobis OOD Triage Gateway")
        
        # Mathematics of Mahalanobis Anomaly Screening:
        # D_M = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
        delta = feature_vector[0] - ood_parameters['mean']
        mahalanobis_dist = np.sqrt(np.dot(np.dot(delta, ood_parameters['inv_covariance']), delta.T))
        
        st.metric(label="Calculated Mahalanobis Deviation Score ($D_M$)", value=f"{mahalanobis_dist:.4f}")
        
        # Evaluation Decision Router Check
        if mahalanobis_dist > ood_threshold:
            # Route to Human Review path
            st.error("🚨 **TRIAGE ACTION REQUIRED: OUT-OF-DISTRIBUTION ANOMALY DETECTED**")
            st.markdown(f"""
            > **System Status**: The feature signature diverges significantly from standard training profiles 
            > ($D_M = {mahalanobis_dist:.2f} > \text{{Threshold}} = {ood_threshold:.2f}$). 
            > 
            > **Action**: Automated prediction has been aborted. This case has been routed to **Human Clinical Review**.
            """)
        else:
            st.success("✅ **IN-DISTRIBUTION PROFILE VERIFIED**")
            st.info("System Status: Sample pattern matches training domain profiles. Routing to the automated Random Forest classifier...")
            
            # --- Pipeline Stage 2: Downstream Model Classification ---
            st.subheader("🎯 Step 3: Random Forest Inference Engine")
            with st.spinner("Running ensemble inference checks..."):
                scaled_features = scaler.transform(feature_vector)

                # Predicted label (could be int or string)
                prediction_label = classifier.predict(scaled_features)[0]

                # Probabilities for each class
                probabilities = classifier.predict_proba(scaled_features)[0]

                # Get the classes array from the classifier
                classes = classifier.classes_

                # Find the index of the predicted label in the classes array
                prediction_idx = list(classes).index(prediction_label)

                # Now you can safely get the confidence
                confidence = probabilities[prediction_idx] * 100

                # Map to human-readable label
                class_label = "Chicken Pox" if prediction_label == 0 else "Measles"

                
                st.markdown(f"### Final Decision: **{class_label}**")
                st.progress(int(confidence))
                st.caption(f"Statistical classification certainty score: {confidence:.2f}%")