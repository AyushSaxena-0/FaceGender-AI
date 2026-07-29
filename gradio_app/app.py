"""
Gradio Web Application for Face Presentation Classifier.
Modern Dark-themed UI with Live Webcam Capture, Image Upload, Multi-Face Selection,
Confidence Meter, Top-2 Probabilities, Grad-CAM/Score-CAM Explainability, and Model Switching.
"""

import os
import yaml
import gradio as gr
import numpy as np

from inference.pipeline import FacePresentationPipeline, DISCLAIMER_TEXT
from utils.logger import setup_logger

logger = setup_logger("GradioApp")

# Instantiate global pipeline
pipeline = FacePresentationPipeline(config_path="configs/config.yaml")


def process_inference(image, face_index, detector_choice, backbone_choice, explainability_choice):
    """
    Main inference handler for Gradio UI.
    """
    if image is None:
        return (
            "<div class='error-box'>⚠️ Please provide an image via Webcam Capture or File Upload.</div>",
            None, None, None, None, None, None
        )

    # Dynamic detector switch
    if detector_choice and detector_choice != pipeline.detector.detector_type:
        pipeline.detector.detector_type = detector_choice
        pipeline.detector._init_detector()

    # Dynamic backbone switch
    if backbone_choice and backbone_choice != pipeline.model.backbone_name:
        try:
            pipeline.config["model"]["backbone"] = backbone_choice
            pipeline.model = FacePresentationPipeline(config_path="configs/config.yaml").model
            logger.info(f"Switched backbone model to: {backbone_choice}")
        except Exception as e:
            logger.warning(f"Could not switch backbone: {e}")

    # Parse face index string "Face #1", "Face #2" -> int index 0, 1
    idx = 0
    if isinstance(face_index, str) and "Face #" in face_index:
        try:
            idx = int(face_index.split("#")[1].split(" ")[0]) - 1
        except Exception:
            idx = 0

    exp_method = "scorecam" if "Score" in str(explainability_choice) else "gradcam"
    res = pipeline.predict(image, face_index=idx, explainability_method=exp_method)

    if not res["success"]:
        error_html = f"""
        <div class="disclaimer-card">
            <span class="warning-icon">⚠️</span>
            <div class="disclaimer-text">
                <strong>{res['disclaimer']}</strong>
            </div>
        </div>
        <div class="error-box">
            <h3>❌ Analysis Error</h3>
            <p>{res['message']}</p>
        </div>
        """
        return error_html, None, None, None, gr.Dropdown(choices=["Face #1"], value="Face #1"), None, None

    # Construct HTML Card
    pred_color = "#3A86FF" if "Masculine" in res["prediction"] else "#FF007F"
    conf = res["confidence_percentage"]
    inf_time = res["inference_time_ms"]
    face_cnt = res["face_count"]
    bbox = res["bounding_box"]

    dropdown_choices = [f"Face #{i+1}" for i in range(face_cnt)]
    selected_choice = f"Face #{idx+1}"

    # Low Confidence Banner (<70%)
    uncertainty_banner = ""
    if res.get("is_uncertain", False):
        uncertainty_banner = f"""
        <div class="uncertainty-banner">
            <span class="warning-icon">⚠️</span>
            <div>
                <strong>PREDICTION UNCERTAIN:</strong> Model confidence ({conf:.1f}%) is below the 70% threshold. Please upload a clearer image.
            </div>
        </div>
        """

    result_html = f"""
    <div class="disclaimer-card">
        <span class="warning-icon">⚖️</span>
        <div class="disclaimer-text">
            <strong>IMPORTANT DISCLAIMER:</strong> {res['disclaimer']}
        </div>
    </div>

    {uncertainty_banner}
    
    <div class="prediction-card">
        <div class="pred-header">
            <span class="pred-badge" style="background-color: {pred_color};">{res['prediction']}</span>
            <span class="time-badge">⚡ {inf_time} ms</span>
        </div>

        <div class="metric-container">
            <div class="metric-label">
                <span>Top Confidence Score</span>
                <span class="metric-value">{conf:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {conf}%; background: linear-gradient(90deg, #4EA8DE, {pred_color});"></div>
            </div>
        </div>

        <div class="top2-container">
            <h4>📊 Top-2 Class Probabilities:</h4>
            <div class="top2-row">
                <span>👨 Masculine-presenting:</span>
                <strong>{res['class_probabilities'].get('Masculine-presenting', 0.0):.1f}%</strong>
            </div>
            <div class="top2-row">
                <span>👩 Feminine-presenting:</span>
                <strong>{res['class_probabilities'].get('Feminine-presenting', 0.0):.1f}%</strong>
            </div>
        </div>

        <div class="meta-info">
            <span>👥 Faces Detected: <strong>{face_cnt}</strong></span>
            <span>📍 Bounding Box: <code>[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</code></span>
        </div>
    </div>
    """

    probs_dict = {
        "Masculine-presenting": res["class_probabilities"]["Masculine-presenting"] / 100.0,
        "Feminine-presenting": res["class_probabilities"]["Feminine-presenting"] / 100.0
    }

    return (
        result_html,
        probs_dict,
        res["annotated_image"],
        res["cropped_face"],
        gr.Dropdown(choices=dropdown_choices, value=selected_choice, visible=face_cnt > 1),
        res["explainability_overlay"],
        res["preprocessed_face"]
    )


# Custom CSS for Dark Theme and Aesthetics
custom_css = """
body, .gradio-container {
    background-color: #0B0F19 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #E2E8F0 !important;
}

.disclaimer-card {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    backdrop-filter: blur(8px);
}

.uncertainty-banner {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid #F59E0B;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    color: #FDE68A;
    font-size: 0.92rem;
}

.warning-icon {
    font-size: 1.6rem;
}

.disclaimer-text {
    font-size: 0.92rem;
    line-height: 1.45;
    color: #FCA5A5;
}

.prediction-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin-top: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.pred-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.pred-badge {
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1.25rem;
    padding: 8px 18px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}

.time-badge {
    background: #0F172A;
    border: 1px solid #334155;
    color: #38BDF8;
    padding: 6px 14px;
    border-radius: 12px;
    font-size: 0.88rem;
    font-weight: 600;
}

.metric-container {
    margin: 20px 0;
}

.metric-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: #94A3B8;
}

.metric-value {
    color: #38BDF8;
    font-weight: 700;
}

.progress-bar-bg {
    width: 100%;
    height: 12px;
    background: #0F172A;
    border-radius: 6px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease-in-out;
}

.top2-container {
    background: #0F172A;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 16px 0;
    border: 1px solid #334155;
}

.top2-container h4 {
    margin: 0 0 10px 0;
    font-size: 0.9rem;
    color: #94A3B8;
}

.top2-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    margin-bottom: 6px;
    color: #CBD5E1;
}

.meta-info {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #64748B;
    border-top: 1px solid #334155;
    padding-top: 14px;
    margin-top: 16px;
}

.error-box {
    background: rgba(225, 29, 72, 0.15);
    border: 1px solid #F43F5E;
    border-radius: 12px;
    padding: 20px;
    color: #FECDD3;
}
"""


def build_gradio_ui():
    """
    Builds the interactive Gradio interface.
    """
    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate"
    ).set(
        body_background_fill="#0B0F19",
        block_background_fill="#1E293B",
        block_border_color="#334155"
    )

    with gr.Blocks(theme=theme, css=custom_css, title="Face Presentation Classifier") as app:
        gr.Markdown(
            """
            # 👤 Face Presentation Classifier
            ### Production Deep Learning Visual Presentation Estimation Engine
            """
        )

        gr.HTML(
            f"""
            <div class="disclaimer-card">
                <span class="warning-icon">⚖️</span>
                <div class="disclaimer-text">
                    <strong>IMPORTANT DISCLAIMER:</strong> {DISCLAIMER_TEXT}
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("📷 Capture Photo (Webcam)"):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            label="Live Webcam Stream"
                        )

                    with gr.TabItem("📁 Upload Image"):
                        file_input = gr.Image(
                            sources=["upload"],
                            type="numpy",
                            label="Upload Image (JPG, PNG, WEBP)"
                        )

                with gr.Accordion("⚙️ Pipeline & Model Configuration", open=False):
                    detector_dropdown = gr.Dropdown(
                        choices=["mediapipe", "retinaface", "opencv"],
                        value=pipeline.detector.detector_type,
                        label="Face Detector Engine"
                    )
                    backbone_dropdown = gr.Dropdown(
                        choices=["efficientnet_b0", "efficientnet_v2_s", "resnet50", "mobilenet_v3_large", "convnext_tiny"],
                        value=pipeline.config.get("model", {}).get("backbone", "efficientnet_b0"),
                        label="Model Backbone Architecture"
                    )
                    explainability_dropdown = gr.Dropdown(
                        choices=["Grad-CAM", "Score-CAM"],
                        value="Grad-CAM",
                        label="Explainability Heatmap Method"
                    )

                face_selector = gr.Dropdown(
                    choices=["Face #1"],
                    value="Face #1",
                    label="Select Target Face (When Multiple Faces Exist)",
                    visible=False
                )

                with gr.Row():
                    analyze_btn = gr.Button("🚀 Analyze Image", variant="primary", scale=2)
                    reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

            # Right Column: Outputs & Analysis
            with gr.Column(scale=6):
                output_card_html = gr.HTML(label="Prediction Card")
                label_scores = gr.Label(label="Top-2 Class Distribution", num_top_classes=2)

                with gr.Tabs():
                    with gr.TabItem("🎯 Bounding Boxes"):
                        annotated_output = gr.Image(label="Detected Face Bounding Boxes", interactive=False)

                    with gr.TabItem("✂️ Cropped Face"):
                        cropped_output = gr.Image(label="Cropped & Eye-Aligned Face", interactive=False)

                    with gr.TabItem("🔥 Explainability Heatmap"):
                        gradcam_output = gr.Image(label="Grad-CAM / Score-CAM Visual Attention Map", interactive=False)

                    with gr.TabItem("🔬 Preprocessed Face"):
                        preprocessed_output = gr.Image(label="CLAHE Brightness Normalized Face", interactive=False)

        # Event Handlers
        active_image = gr.State(None)

        def set_webcam(img):
            return img

        def set_upload(img):
            return img

        webcam_input.change(set_webcam, inputs=[webcam_input], outputs=[active_image])
        file_input.change(set_upload, inputs=[file_input], outputs=[active_image])

        analyze_btn.click(
            fn=process_inference,
            inputs=[
                active_image,
                face_selector,
                detector_dropdown,
                backbone_dropdown,
                explainability_dropdown
            ],
            outputs=[
                output_card_html,
                label_scores,
                annotated_output,
                cropped_output,
                face_selector,
                gradcam_output,
                preprocessed_output
            ]
        )

        def clear_all():
            return None, None, "", None, None, None, gr.Dropdown(choices=["Face #1"], value="Face #1", visible=False), None, None

        reset_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                webcam_input,
                file_input,
                output_card_html,
                label_scores,
                annotated_output,
                cropped_output,
                face_selector,
                gradcam_output,
                preprocessed_output
            ]
        )

    return app


if __name__ == "__main__":
    app = build_gradio_ui()
    server_name = pipeline.config.get("gradio", {}).get("server_name", "0.0.0.0")
    server_port = pipeline.config.get("gradio", {}).get("server_port", 7860)
    app.queue().launch(server_name=server_name, server_port=server_port, share=False)
