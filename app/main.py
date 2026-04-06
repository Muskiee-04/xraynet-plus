import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import torch
import sys
import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from src.data.image_loading import load_image_bytes_to_rgb
    from src.data.preprocessing import CXRPreprocessor
    from src.inference.torch_inference import TorchCXRInference
    from src.utils.helpers import overlay_heatmap_on_image
    from database import DatabaseManager
    from report_generator import PDFReportGenerator, generate_report

    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORT_SUCCESS = False

try:
    from src.ai.gemini_service import (
        chat_reply_with_history,
        configure_from_key,
        generate_case_interpretation,
        generate_heatmap_explanation_plain_language,
        generate_radiology_question_suggestions,
        is_configured,
        multimodal_compare_image,
    )

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration with robot theme
st.set_page_config(
    page_title="ChestRay Gemini — CXR screening + Gemini copilot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cartoon robot theme with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Floating particles animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
    }
    
    /* Robot container */
    .robot-container {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    /* Robot SVG */
    .robot-svg {
        font-size: 180px;
        animation: pulse 2s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.2));
    }
    
    /* Upload zone */
    .upload-zone {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        border: 4px dashed white;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border: 3px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
    }
    
    /* Icon badges */
    .icon-badge {
        display: inline-block;
        font-size: 3rem;
        margin: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Metric cards with icons */
    .metric-icon-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    .metric-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(245, 87, 108, 0.6) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px 15px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Floating animation for icons */
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Speech bubble */
    .speech-bubble {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        font-size: 1.1rem;
        color: #333;
    }
    
    .speech-bubble:before {
        content: '';
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        border: 15px solid transparent;
        border-top-color: white;
    }
</style>
""", unsafe_allow_html=True)

class XrayNetPlusApp:
    def __init__(self):
        self.preprocessor = CXRPreprocessor()
        self.db_manager = DatabaseManager()  # Fixed: Now DatabaseManager is defined
        self.report_generator = PDFReportGenerator()
        
        # Initialize model (will be loaded on first use)
        self.inference_engine = None
        self.model_loaded = False
        
        # Class names
        self.class_names = {
            0: "Tuberculosis",
            1: "Pneumonia", 
            2: "COVID-19",
            3: "No Findings"
        }
        
    def load_model(self):
        """Load PyTorch EfficientNet + Grad-CAM++ pipeline."""
        if not self.model_loaded:
            try:
                # Use same priority as TorchCXRInference: xraynet_unified_finetuned.pth → xraynet_plus.pth → …
                # or ``XRAYNET_WEIGHTS`` (checked inside TorchCXRInference).
                self.inference_engine = TorchCXRInference(weights_path=None)
                self.model_loaded = True
                loaded = getattr(self.inference_engine, "weights_loaded_from", None)
                if loaded:
                    st.success(f"✅ Model weights loaded from `{loaded}`")
                else:
                    st.info(
                        "ℹ️ No checkpoint found in `models/saved/`. If ImageNet download failed (e.g. SSL on macOS), "
                        "the backbone is random until you run `python scripts/init_demo_model.py` on a network with "
                        "valid certs, or add weights under `models/saved/` (e.g. `xraynet_unified_finetuned.pth`)."
                    )
                return True
            except Exception as e:
                st.error(f"❌ Model Error: {e}")
                return False
        return True
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'patient_data' not in st.session_state:
            st.session_state.patient_data = {}
        if 'reports_generated' not in st.session_state:
            st.session_state.reports_generated = False
        if 'show_admin' not in st.session_state:
            st.session_state.show_admin = False
        if 'gemini_chat_history' not in st.session_state:
            st.session_state.gemini_chat_history = []
        if 'gemini_narrative' not in st.session_state:
            st.session_state.gemini_narrative = ""
        if 'gemini_model_name' not in st.session_state:
            st.session_state.gemini_model_name = ""

    def _sidebar_gemini_settings(self):
        """Optional Google AI Studio API key (free tier) for Gemini features."""
        if not GEMINI_AVAILABLE:
            return
        with st.sidebar.expander("✨ Google Gemini (optional)", expanded=False):
            st.caption("Get a key at [Google AI Studio](https://aistudio.google.com/apikey). Stored only in this browser session unless you use env/secrets.")
            key = st.text_input("GEMINI_API_KEY", type="password", key="gemini_api_key_sidebar")
            if key and key.strip():
                try:
                    configure_from_key(key.strip())
                    st.caption("API key active for this browser session.")
                except Exception as ex:
                    st.error(str(ex))
            env_ok = is_configured()
            if env_ok and not (key and key.strip()):
                st.info("Using GEMINI_API_KEY from environment or Streamlit secrets.")
            import os

            st.session_state.gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            st.caption(f"Model: `{st.session_state.gemini_model_name}` — override with env `GEMINI_MODEL`.")

    def sidebar_upload(self):
        """Friendly sidebar for file upload"""
        with st.sidebar:
            # Robot header
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <div style='font-size: 80px;'>🤖</div>
                <h2 style='color: white; margin: 0;'>RoboRadiology</h2>
                <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0;'>Your Friendly AI Assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Admin toggle
            if st.button("🏥 Admin Dashboard", use_container_width=True):
                st.session_state.show_admin = not st.session_state.show_admin
            
            st.markdown("---")
            
            # Patient Information
            st.markdown("### 👤 Patient Details")
            
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID", value="PAT_001")
            with col2:
                age = st.number_input("Age", min_value=0, max_value=120, value=45)
            
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            clinical_notes = st.text_area("Clinical Notes", 
                                        placeholder="Enter symptoms and observations...",
                                        height=100)
            
            st.markdown("---")
            
            # File Upload
            st.markdown("### 📸 Upload X-Rays")
            uploaded_files = st.file_uploader(
                "Drag and drop chest X-rays (images or DICOM)",
                type=["png", "jpg", "jpeg", "dcm", "webp", "tif", "tiff", "bmp"],
                accept_multiple_files=True,
            )
            
            st.markdown("---")

            self._sidebar_gemini_settings()

            st.markdown("---")
            
            # Friendly info
            st.markdown("""
            <div style='background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 15px;'>
                <div style='font-size: 2rem; text-align: center;'>🔬</div>
                <h4 style='color: white; text-align: center; margin: 0.5rem 0;'>I Can Detect:</h4>
                <p style='color: white; margin: 0.5rem 0; text-align: center;'>
                🦠 Tuberculosis<br>
                🫁 Pneumonia<br>
                😷 COVID-19<br>
                ✅ Healthy Lungs
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return uploaded_files, {
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'clinical_notes': clinical_notes,
                'timestamp': st.session_state.timestamp
            }
    
    def process_single_image(self, uploaded_file, patient_data):
        """Process a single uploaded file"""
        try:
            raw = uploaded_file.getvalue()
            name = getattr(uploaded_file, "name", "") or "upload"
            image_np = load_image_bytes_to_rgb(raw, name)

            image_tensor, original_image = self.preprocessor.preprocess_for_inference(image_np)
            
            if image_tensor is not None:
                if self.load_model():
                    detailed_pred = self.inference_engine.get_detailed_prediction(
                        image_tensor, original_image
                    )
                    heatmap_image = detailed_pred.pop("heatmap_rgb")
                    return {
                        "original_image": original_image,
                        "processed_image": image_tensor,
                        "prediction": detailed_pred,
                        "heatmap": heatmap_image,
                        "filename": uploaded_file.name,
                    }
                    
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            return None
        
    def display_results(self, results):
        """Display results with icons and visual elements"""
        if not results:
            return
            
        # Summary metrics with icons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>📊</div>
                <div class='metric-value'>{len(results)}</div>
                <div class='metric-label'>Images Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_confidence = np.mean([r['prediction']['confidence'] for r in results])
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>🎯</div>
                <div class='metric-value'>{avg_confidence:.0%}</div>
                <div class='metric-label'>Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            primary_diagnosis = max(set([r['prediction']['class_name'] for r in results]), 
                                  key=[r['prediction']['class_name'] for r in results].count)
            diagnosis_icon = "🦠" if "Tuberculosis" in primary_diagnosis else "🫁" if "Pneumonia" in primary_diagnosis else "😷" if "COVID" in primary_diagnosis else "✅"
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>{diagnosis_icon}</div>
                <div class='metric-value' style='font-size: 1.3rem;'>{primary_diagnosis}</div>
                <div class='metric-label'>Primary Finding</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display each result
        for result in results:
            self.display_single_result(result)
    
    def display_single_result(self, result):
        """Display single result with enhanced visuals"""
        st.markdown(f"""
        <div class='result-card'>
            <h3>🖼️ {result['filename']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📸 Original Image", "🔥 AI Heatmap", "🎨 Combined View"])
        
        with tab1:
            display_img = result['original_image']
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            elif display_img.shape[2] == 4:
                display_img = display_img[:, :, :3]
            if display_img.dtype != np.uint8:
                display_img = (display_img * 255).astype(np.uint8)
            st.image(display_img, use_container_width=True, caption="Original Chest X-Ray")
        
        with tab2:
            st.image(result['heatmap'], use_container_width=True, caption="AI Attention Areas")
        
        with tab3:
            original_rgb = result["original_image"]
            if len(original_rgb.shape) == 2:
                original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
            elif original_rgb.shape[2] == 4:
                original_rgb = original_rgb[:, :, :3]
            if original_rgb.dtype != np.uint8:
                original_rgb = (np.clip(original_rgb, 0, 1) * 255).astype(np.uint8)
            overlay = overlay_heatmap_on_image(original_rgb, result["heatmap"], alpha=0.45)
            st.image(overlay, use_container_width=True, caption="Combined Analysis View")
        
        # Diagnosis with icon
        pred = result['prediction']
        diagnosis_icon = "🦠" if "Tuberculosis" in pred['class_name'] else "🫁" if "Pneumonia" in pred['class_name'] else "😷" if "COVID" in pred['class_name'] else "✅"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;'>
            <div style='font-size: 3rem; text-align: center;'>{diagnosis_icon}</div>
            <h2 style='text-align: center; margin: 0.5rem 0;'>{pred['class_name']}</h2>
            <p style='text-align: center; font-size: 1.5rem; margin: 0;'>{pred['confidence']:.1%} Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed probabilities
        with st.expander("📊 Detailed Analysis", expanded=True):
            for class_name, prob in pred['probabilities'].items():
                prob_value = float(prob)
                class_icon = "🦠" if "Tuberculosis" in class_name else "🫁" if "Pneumonia" in class_name else "😷" if "COVID" in class_name else "✅"
                col1, col2, col3 = st.columns([1, 5, 1])
                with col1:
                    st.markdown(f"<div style='font-size: 2rem;'>{class_icon}</div>", unsafe_allow_html=True)
                with col2:
                    st.write(f"**{class_name}**")
                    st.progress(prob_value)
                with col3:
                    st.write(f"**{prob_value:.0%}**")
        
        # Recommendations (class-specific guidance + prevention)
        st.info(f"💡 **Summary:** {pred['recommendation']}")
        clin = pred.get("clinical_steps")
        prev = pred.get("prevention")
        if not clin or not prev:
            from src.utils.cxr_recommendations import get_recommendation_detail

            d = get_recommendation_detail(pred["class_name"])
            clin = clin or d["clinical_steps"]
            prev = prev or d["prevention"]
        with st.expander("📋 Clinical next steps & follow-up", expanded=True):
            for line in clin:
                st.markdown(f"- {line}")
        with st.expander("🛡️ Prevention & healthy habits", expanded=False):
            for line in prev:
                st.markdown(f"- {line}")
        st.caption(
            "Educational information only—not a medical diagnosis. "
            "Always follow advice from your qualified healthcare professional."
        )

    def display_gemini_copilot(self, patient_data, results):
        """Google Gemini: interpretation, Q&A, optional vision (user consent)."""
        st.markdown("---")
        st.markdown("## ✨ Gemini clinical copilot (educational)")
        st.caption(
            "Uses Google’s API when a key is set. Outputs are not diagnoses; do not upload identifiable PHI without authorization."
        )

        if not GEMINI_AVAILABLE:
            st.warning("Install `google-generativeai` (see requirements.txt) to enable Gemini features.")
            return

        if not is_configured():
            st.info("Add **GEMINI_API_KEY** in the sidebar expander, environment, or `.streamlit/secrets.toml`.")
            return

        tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
            ["Case narrative", "Ask a radiologist", "Heatmap explainer", "Chat", "Vision (consent)"]
        )

        with tab_a:
            if st.button("Generate structured case narrative", key="gemini_narr_btn"):
                with st.spinner("Calling Gemini…"):
                    try:
                        text = generate_case_interpretation(patient_data, results)
                        st.session_state.gemini_narrative = text
                        st.success("Saved for optional PDF appendix.")
                    except Exception as ex:
                        st.error(str(ex))
                        text = ""
                if st.session_state.get("gemini_narrative"):
                    st.markdown(st.session_state.gemini_narrative)
            elif st.session_state.get("gemini_narrative"):
                st.markdown(st.session_state.gemini_narrative)

        with tab_b:
            if st.button("Suggest questions for your clinician", key="gemini_q_btn"):
                with st.spinner("Calling Gemini…"):
                    try:
                        st.session_state.gemini_questions = generate_radiology_question_suggestions(
                            patient_data, results
                        )
                    except Exception as ex:
                        st.error(str(ex))
            if st.session_state.get("gemini_questions"):
                st.markdown(st.session_state.gemini_questions)

        with tab_c:
            names = [r.get("filename", f"image_{i}") for i, r in enumerate(results)]
            pick = st.selectbox("Image for explainer", range(len(names)), format_func=lambda i: names[i])
            if st.button("Explain heatmaps in plain language", key="gemini_hm_btn"):
                with st.spinner("Calling Gemini…"):
                    try:
                        st.session_state.gemini_heatmap_expl = generate_heatmap_explanation_plain_language(
                            patient_data, results[pick]
                        )
                    except Exception as ex:
                        st.error(str(ex))
            if st.session_state.get("gemini_heatmap_expl"):
                st.markdown(st.session_state.gemini_heatmap_expl)

        with tab_d:
            if st.button("Clear chat history", key="gemini_chat_clear"):
                st.session_state.gemini_chat_history = []
            for turn in st.session_state.gemini_chat_history:
                role = turn.get("role", "")
                if role == "user":
                    st.chat_message("user").write(turn.get("parts", ""))
                else:
                    st.chat_message("assistant").write(turn.get("parts", ""))
            msg = st.chat_input("Ask about this case (educational only)…", key="gemini_chat_input")
            if msg:
                st.session_state.gemini_chat_history.append({"role": "user", "parts": msg})
                with st.spinner("Gemini…"):
                    try:
                        reply = chat_reply_with_history(
                            msg,
                            patient_data,
                            results,
                            st.session_state.gemini_chat_history[:-1],
                        )
                    except Exception as ex:
                        reply = f"Error: {ex}"
                st.session_state.gemini_chat_history.append({"role": "model", "parts": reply})
                st.rerun()

        with tab_e:
            st.warning(
                "Sending images to Google leaves your network. Do not use for identifiable patient data without compliance review."
            )
            names = [r.get("filename", f"image_{i}") for i, r in enumerate(results)]
            idx = st.selectbox("Image to send", range(len(names)), format_func=lambda i: names[i], key="gemini_vis_idx")
            consent = st.checkbox("I understand and consent to sending this image to the Gemini API.", key="gemini_vis_ok")
            extra = st.text_area("Optional focus for the model", value="Educational read consistent with class probabilities.")
            if consent and st.button("Run vision-assisted commentary", key="gemini_vis_btn"):
                img = results[idx]["original_image"]
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                ok, enc = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not ok:
                    st.error("Could not encode image.")
                else:
                    with st.spinner("Gemini vision…"):
                        try:
                            st.session_state.gemini_vision = multimodal_compare_image(
                                enc.tobytes(),
                                "image/png",
                                patient_data,
                                results,
                                user_prompt=extra or "Educational commentary.",
                            )
                        except Exception as ex:
                            st.session_state.gemini_vision = f"Error: {ex}"
            if st.session_state.get("gemini_vision"):
                st.markdown(st.session_state.gemini_vision)
    
    def generate_report_section(self, patient_data, results):
        """Report generation section"""
        st.markdown("---")
        st.markdown("### 📄 Generate Medical Report")
        include_gemini = False
        if GEMINI_AVAILABLE and st.session_state.get("gemini_narrative"):
            include_gemini = st.checkbox(
                "Append Gemini case narrative to PDF (if generated above)",
                value=False,
                key="pdf_include_gemini",
            )

        if st.button("🖨️ Create Comprehensive Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating your report..."):
                try:
                    # Store examination data first
                    examination_id = self.db_manager.store_examination(patient_data, results)
                    
                    if examination_id:
                        st.success(f"✅ Examination data stored (ID: {examination_id})")
                    
                    appendix = (
                        st.session_state.get("gemini_narrative")
                        if include_gemini and st.session_state.get("gemini_narrative")
                        else None
                    )
                    pdf_buffer = self.report_generator.generate_report(
                        patient_data, results, gemini_appendix=appendix
                    )
                    
                    st.success("✅ Report Generated Successfully!")
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"Medical_Report_{patient_data['patient_id']}_{patient_data['timestamp']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.session_state.reports_generated = True
                    
                except Exception as e:
                    st.error(f"❌ Report generation failed: {e}")
    
    def admin_dashboard(self):
        """Admin dashboard for viewing stored data"""
        st.markdown("---")
        st.markdown("## 🏥 Admin Dashboard")
        
        if st.button("📊 View Database Statistics", use_container_width=True):
            stats = self.db_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", stats['total_patients'])
            with col2:
                st.metric("Total Examinations", stats['total_examinations'])
            with col3:
                st.metric("Total Images Analyzed", stats['total_images'])
            
            # Common findings
            st.subheader("Most Common Findings")
            for finding, count in stats['common_findings']:
                st.write(f"**{finding}**: {count} cases")
        
        if st.button("👥 View All Patients", use_container_width=True):
            patients = self.db_manager.get_all_patients()
            
            if patients:
                st.subheader("Patient Database")
                for patient in patients:
                    with st.expander(f"Patient: {patient['patient_id']} - {patient.get('name', 'N/A')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Age**: {patient['age']}")
                            st.write(f"**Gender**: {patient['gender']}")
                        with col2:
                            st.write(f"**Examinations**: {patient['total_examinations']}")
                            st.write(f"**Last Visit**: {patient['last_examination']}")
                        
                        # Show examinations for this patient
                        exams = self.db_manager.get_patient_examinations(patient['patient_id'])
                        for exam in exams:
                            st.write(f"📅 Examination on {exam['created_at']}: {exam['primary_finding']} ({exam['average_confidence']:.1%} confidence)")
            
            else:
                st.info("No patients found in database.")
        
        # Data export options
        st.subheader("Data Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Patients to CSV"):
                filename = self.db_manager.export_to_csv('patients')
                st.success(f"Patients exported to {filename}")
        
        with col2:
            if st.button("💾 Create Database Backup"):
                backup_path = self.db_manager.backup_database()
                if backup_path:
                    st.success(f"Database backed up to {backup_path}")
                else:
                    st.error("Backup failed")
    
    def display_welcome_screen(self):
        """Welcome screen with cartoon robot"""
        # Big friendly robot
        st.markdown("""
        <div class='robot-container'>
            <div class='robot-svg'>🤖</div>
            <h1 style='color: #667eea; margin: 1rem 0;'>Hello! I'm ChestRay Gemini</h1>
            <div class='speech-bubble'>
                <p style='margin: 0;'><strong>Hi there!</strong> 👋 Local EfficientNet screening plus optional Google Gemini narratives, chat, and (with consent) vision—upload a chest X-ray to begin.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Scientific animations background
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <span class='icon-badge floating' style='animation-delay: 0s;'>🔬</span>
            <span class='icon-badge floating' style='animation-delay: 0.5s;'>🧬</span>
            <span class='icon-badge floating' style='animation-delay: 1s;'>⚗️</span>
            <span class='icon-badge floating' style='animation-delay: 1.5s;'>🧪</span>
            <span class='icon-badge floating' style='animation-delay: 2s;'>🔭</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload zone
        st.markdown("""
        <div class='upload-zone'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📤</div>
            <div>Upload Patient Details & X-Ray Images</div>
            <div style='font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;'>Use the sidebar to get started! ➡️</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>🎯</div>
                <h3 style='color: #667eea;'>Accurate Detection</h3>
                <p>AI-powered analysis with confidence scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>🔍</div>
                <h3 style='color: #667eea;'>Visual Explanations</h3>
                <p>See exactly where AI detects abnormalities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>📊</div>
                <h3 style='color: #667eea;'>Gemini copilot</h3>
                <p>Optional AI narratives, chat, and teaching prompts</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h4 style='color: #667eea;'>📝 Step 1: Enter Patient Info</h4>
                <p>Fill in patient details in the sidebar</p>
                
                <h4 style='color: #667eea; margin-top: 1.5rem;'>📸 Step 2: Upload Images</h4>
                <p>Drag and drop chest X-ray images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h4 style='color: #667eea;'>🔬 Step 3: Review Analysis</h4>
                <p>I'll analyze and show visual heatmaps</p>
                
                <h4 style='color: #667eea; margin-top: 1.5rem;'>📄 Step 4: Get Report</h4>
                <p>Download comprehensive PDF report</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Disclaimer
        st.info("⚠️ **Important:** This AI assistant is designed to support medical professionals. All findings should be reviewed by qualified healthcare providers.")

    def run(self):
        """Main application runner"""
        self.init_session_state()

        if GEMINI_AVAILABLE and not is_configured():
            try:
                sec = getattr(st, "secrets", None)
                if sec and sec.get("GEMINI_API_KEY"):
                    configure_from_key(str(sec["GEMINI_API_KEY"]))
            except (FileNotFoundError, KeyError, AttributeError, RuntimeError):
                pass
        
        # Show admin dashboard if toggled
        if st.session_state.show_admin:
            self.admin_dashboard()
            return
        
        uploaded_files, patient_data = self.sidebar_upload()
        st.session_state.patient_data = patient_data
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
            
            st.markdown("---")
            st.markdown("### 🔄 Analyzing Images...")
            
            results = []
            progress_bar = st.progress(0, text="Starting AI analysis...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                result = self.process_single_image(uploaded_file, patient_data)
                if result:
                    results.append(result)
                progress_value = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress_value, text=f"Analyzed {i+1}/{len(uploaded_files)} images")
            
            progress_bar.empty()
            
            if results:
                st.session_state.predictions = results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                self.display_results(results)
                self.display_gemini_copilot(patient_data, results)
                self.generate_report_section(patient_data, results)
            else:
                st.error("❌ No images were successfully processed. Please check your files and try again.")
        else:
            self.display_welcome_screen()

# Run the app
if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        st.stop()
    app = XrayNetPlusApp()
    app.run()