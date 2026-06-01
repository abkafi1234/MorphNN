import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial import distance

# ---------------------------------------------------------
# Re-importing your exact feature extraction architecture
# ---------------------------------------------------------
class ProductionInferencePipeline:
    def __init__(self, cnn_model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dynamic CNN backbone loader matching your benchmark setup
        if cnn_model_name == 'EfficientNet-B0':
            self.cnn_extractor = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        elif cnn_model_name == 'ResNet18':
            self.cnn_extractor = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2])
        elif cnn_model_name == 'ResNet34':
            self.cnn_extractor = nn.Sequential(*list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-2])
        elif cnn_model_name == 'MobileNetV2':
            self.cnn_extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        elif cnn_model_name == 'MobileNetV3':
            self.cnn_extractor = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).features
        elif cnn_model_name == 'ShuffleNetV2':
            self.cnn_extractor = nn.Sequential(*list(models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT).children())[:-1])
        elif cnn_model_name == 'SqueezeNet':
            self.cnn_extractor = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).features
            
        self.cnn_extractor.to(self.device)
        self.cnn_extractor.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classic_names = ['Lesion Count', 'Avg Area', 'Std Dev Area', 'Avg Circularity', 
                              'Sparsity Score', 'Confluence Ratio', 'Avg Hue', 'Avg Saturation']

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

    def extract_features_from_bytes(self, opencv_img):
        img_512 = cv2.resize(opencv_img, (512, 512))
        smoothed = cv2.bilateralFilter(img_512, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
        
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

        wb_img = self.apply_gray_world_white_balance(img_512)
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
        classic_features = [len(centroids), np.mean(areas) if areas else 0, std_area, 
                            avg_circularity, sparsity_score, confluence_ratio, avg_hue, avg_saturation]

        cnn_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(cnn_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.cnn_extractor(input_tensor)
            pooled = self.pool(features)
            cnn_features_raw = pooled.flatten().cpu().numpy()

        return classic_features, cnn_features_raw

# ---------------------------------------------------------
# Streamlit UI Render Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Exanthem Hybrid Diagnostics App", layout="wide")

st.title("🔬 Clinical Exanthem Hybrid Diagnostic Portal")
st.markdown("This system utilizes your top-performing benchmarked configuration pairing handcrafted structural metrics with deep visual feature spaces.")

MODEL_DIR = './best_model'

@st.cache_resource
def load_production_pipeline():
    """Reads pickled models and configures system architecture securely once."""
    if not os.path.exists(f"{MODEL_DIR}/metadata.pkl"):
        return None, None, None, None
        
    with open(f"{MODEL_DIR}/metadata.pkl", 'rb') as f: metadata = pickle.load(f)
    with open(f"{MODEL_DIR}/classifier_model.pkl", 'rb') as f: classifier = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f: scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/pca.pkl", 'rb') as f: pca = pickle.load(f)
    
    # Instantiate the wrapper utilizing the specific framework winner
    pipeline_wrapper = ProductionInferencePipeline(metadata['best_cnn'])
    return pipeline_wrapper, classifier, scaler, pca

pipeline, clf, scaler, pca = load_production_pipeline()

if pipeline is None:
    st.error("🚨 Active model deployment binaries not found in `./best_model/`. Run your benchmark script completely first to build the production assets.")
else:
    # Sidebar details panel
    with st.sidebar:
        st.subheader("System Deployment Parameters")
        st.info(f"**Optimal Backbone:** {pipeline.cnn_extractor.__class__.__name__ if hasattr(pipeline, 'cnn_extractor') else 'Configured Neural Net'}")
        with open(f"{MODEL_DIR}/metadata.pkl", 'rb') as f: meta = pickle.load(f)
        st.success(f"**Classifier Core:** {meta['classifier_name']}")
        st.markdown("---")
        st.markdown("**Hybrid Dimensions:**\n* Handcrafted Spatiotemporal: 8 dimensions\n* PCA-Compressed Visual: 10 dimensions")

    # Image upload center panel
    uploaded_file = st.file_uploader("Upload an exanthem lesion clinical image scan (JPG/PNG format)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption="Uploaded Diagnostic Asset", use_container_width=True)
            
        with col2:
            with st.spinner("Extracting multi-tier feature representations..."):
                # Extract the features using the underlying operational pipeline logic
                classic_feats, cnn_raw = pipeline.extract_features_from_bytes(opencv_image)
                
                # Apply structural scaling transformation layers sequentially
                cnn_pca = pca.transform(cnn_raw.reshape(1, -1))
                combined_vector = np.hstack((np.array(classic_feats).reshape(1, -1), cnn_pca))
                final_input_scaled = scaler.transform(combined_vector)
                
                # Make prediction
                prediction = clf.predict(final_input_scaled)[0]
                
                # Display Results callout
                st.subheader("Diagnostic Evaluation Result")
                st.metric(label="Predicted Diagnostic Class", value=str(prediction).upper())
                
                if hasattr(clf, "predict_proba"):
                    probabilities = clf.predict_proba(final_input_scaled)[0]
                    classes = clf.classes_
                    st.markdown("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame({'Condition': classes, 'Confidence': probabilities})
                    prob_df['Confidence'] = prob_df['Confidence'].map(lambda x: f"{x*100:.2f}%")
                    st.table(prob_df)

            # Display explainability metrics
            st.subheader("Handcrafted Morphological Statistics")
            metrics_df = pd.DataFrame({
                'Feature Type': pipeline.classic_names,
                'Extracted Value': [f"{val:.4f}" for val in classic_feats]
            })
            st.dataframe(metrics_df, use_container_width=True)