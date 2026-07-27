import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

# ==========================================================
# Load Scaler and Model
# ==========================================================

try:
    scaler = joblib.load("breast_cancer_scaler.pkl")
    deployed_nn = tf.keras.models.load_model("breast_cancer_model.h5")
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    scaler = None
    deployed_nn = None

# ==========================================================
# Prediction Function
# ==========================================================

def predict_cancer(*features):

    values = list(features)

    if any(v is None for v in values):
        return "❌ Please fill in all 30 medical measurements."

    try:
        float_values = [float(v) for v in values]
    except Exception as e:
        return f"❌ Invalid Input\n\n{e}"

    if deployed_nn is None or scaler is None:
        return "❌ Model or Scaler not loaded."

    try:
        input_array = np.array([float_values])

        scaled_input = scaler.transform(input_array)

        prediction = deployed_nn.predict(scaled_input, verbose=0)[0][0]

        if prediction >= 0.5:
            return (
                f"🟢 BENIGN\n\n"
                f"Confidence : {prediction:.2%}\n\n"
                "The tumor appears to be non-cancerous."
            )
        else:
            return (
                f"🔴 MALIGNANT\n\n"
                f"Confidence : {(1-prediction):.2%}\n\n"
                "High risk detected. Please consult an oncologist."
            )

    except Exception:
        return traceback.format_exc()

# ==========================================================
# User Interface
# ==========================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🔬 Breast Cancer Detection System")
    gr.Markdown("### Deep Learning Based Prediction")

    inputs = []

    with gr.Tabs():

        with gr.Tab("Mean Metrics"):

            with gr.Row():

                with gr.Column():
                    for label in [
                        "Mean Radius",
                        "Mean Texture",
                        "Mean Perimeter",
                        "Mean Area",
                        "Mean Smoothness",
                    ]:
                        inputs.append(gr.Number(label=label))

                with gr.Column():
                    for label in [
                        "Mean Compactness",
                        "Mean Concavity",
                        "Mean Concave Points",
                        "Mean Symmetry",
                        "Mean Fractal Dimension",
                    ]:
                        inputs.append(gr.Number(label=label))

        with gr.Tab("Error Metrics"):

            with gr.Row():

                with gr.Column():
                    for label in [
                        "Radius Error",
                        "Texture Error",
                        "Perimeter Error",
                        "Area Error",
                        "Smoothness Error",
                    ]:
                        inputs.append(gr.Number(label=label))

                with gr.Column():
                    for label in [
                        "Compactness Error",
                        "Concavity Error",
                        "Concave Points Error",
                        "Symmetry Error",
                        "Fractal Dimension Error",
                    ]:
                        inputs.append(gr.Number(label=label))

        with gr.Tab("Worst Metrics"):

            with gr.Row():

                with gr.Column():
                    for label in [
                        "Worst Radius",
                        "Worst Texture",
                        "Worst Perimeter",
                        "Worst Area",
                        "Worst Smoothness",
                    ]:
                        inputs.append(gr.Number(label=label))

                with gr.Column():
                    for label in [
                        "Worst Compactness",
                        "Worst Concavity",
                        "Worst Concave Points",
                        "Worst Symmetry",
                        "Worst Fractal Dimension",
                    ]:
                        inputs.append(gr.Number(label=label))

    result = gr.Textbox(
        label="Prediction Result",
        lines=5,
        interactive=False,
    )

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict", variant="primary")
        clear_btn = gr.ClearButton(inputs + [result])

    predict_btn.click(
        fn=predict_cancer,
        inputs=inputs,
        outputs=result,
    )

    gr.Markdown(
        """
---
### 👨‍💻 Developer
**Created by: Pavni Jain**
"""
    )

# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
