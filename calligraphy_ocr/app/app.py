"""
Gradio demo for the ECE 693 Arabic calligraphy pipeline.

Stage 1: ResNet18 local-global fusion → calligraphy style.
Stage 2: Qwen-VL OCR backend → Arabic text transcription, optionally
         style-conditioned via a prompt prefix.

Run:
    /home/ahmad/anaconda3/envs/calligraphy_ocr/bin/python \
        calligraphy_ocr/app/app.py

Optional env:
    RESNET18_CKPT=/path/to/style_classifier.pth   # plug in the trained Stage 1
    OCR_DEFAULT_MODEL="Qari-OCR v0.3 (handwriting-aware)"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from style_classifier import StyleClassifier, CLASSES  # noqa: E402
from ocr_backend import OCRBackend, MODELS as OCR_MODELS, DEFAULT_MODEL  # noqa: E402

# Lazy globals — initialized on first call to avoid blocking app boot.
_classifier: StyleClassifier | None = None
_ocr: OCRBackend | None = None


def get_classifier() -> StyleClassifier:
    global _classifier
    if _classifier is None:
        _classifier = StyleClassifier()
    return _classifier


def get_ocr() -> OCRBackend:
    global _ocr
    if _ocr is None:
        _ocr = OCRBackend()
    return _ocr


# ── CSS ──────────────────────────────────────────────────────────────────────
# Manuscript-inspired palette: deep teal ink, illumination gold, aged
# parchment, deep maroon accent. Arabic typography via Amiri (Google Fonts).
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #0d3d3a;
    --ink-dark: #082624;
    --gold: #c9a961;
    --gold-soft: #e3cf9a;
    --parchment: #f5efe0;
    --ivory: #faf6ec;
    --maroon: #6e2a3e;
    --shadow: rgba(13,61,58,0.08);
}

.gradio-container {
    background:
        radial-gradient(circle at 20% 10%, rgba(201,169,97,0.08) 0%, transparent 40%),
        radial-gradient(circle at 80% 90%, rgba(110,42,62,0.05) 0%, transparent 40%),
        linear-gradient(180deg, #faf6ec 0%, #f0e8d2 100%);
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--ink-dark);
    min-height: 100vh;
}

/* Subtle 8-fold-star geometric pattern at very low opacity */
.gradio-container::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><g fill='none' stroke='%23c9a961' stroke-width='0.6' opacity='0.18'><polygon points='40,8 48,32 72,40 48,48 40,72 32,48 8,40 32,32'/><polygon points='40,16 45.5,34.5 64,40 45.5,45.5 40,64 34.5,45.5 16,40 34.5,34.5' transform='rotate(22.5 40 40)'/></g></svg>");
    background-repeat: repeat;
    opacity: 0.7;
}

.gradio-container > * { position: relative; z-index: 1; }

/* Header banner */
#title-banner {
    text-align: center;
    padding: 32px 16px 24px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, rgba(13,61,58,0.04), rgba(201,169,97,0.07));
    border-bottom: 1px solid var(--gold-soft);
    border-radius: 0 !important;
}
#title-banner .ornament {
    color: var(--gold); font-size: 22px; letter-spacing: 16px;
    margin-bottom: 8px; user-select: none;
}
#title-banner h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px; font-weight: 600;
    color: var(--ink-dark);
    letter-spacing: 0.02em;
    margin: 4px 0 6px;
}
#title-banner .arabic-title {
    font-family: 'Amiri', serif;
    font-size: 30px; font-weight: 700;
    color: var(--maroon);
    direction: rtl; margin: 0 0 10px;
}
#title-banner .subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic; font-size: 17px;
    color: var(--ink); opacity: 0.85;
}

/* Card-style panels */
.panel {
    background: var(--ivory) !important;
    border: 1px solid var(--gold-soft) !important;
    border-radius: 4px !important;
    padding: 20px !important;
    box-shadow: 0 2px 12px var(--shadow) !important;
}
.panel-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px; font-weight: 600;
    color: var(--ink-dark);
    border-bottom: 1px solid var(--gold-soft);
    padding-bottom: 8px; margin-bottom: 16px;
}
.panel-title::before { content: "❋"; color: var(--gold); margin-right: 10px; }

/* Buttons */
button.primary, button[variant="primary"], .gr-button-primary {
    background: linear-gradient(135deg, var(--ink), var(--ink-dark)) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--gold) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    border-radius: 3px !important;
    transition: all 0.2s ease !important;
}
button.primary:hover, button[variant="primary"]:hover, .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--maroon), var(--ink-dark)) !important;
    border-color: var(--gold-soft) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(13,61,58,0.18) !important;
}

/* Arabic output zone */
.arabic-output {
    direction: rtl; text-align: right;
    font-family: 'Amiri', 'Scheherazade New', serif !important;
    font-size: 28px; line-height: 1.9;
    color: var(--ink-dark);
    background: linear-gradient(180deg, #fdfaef, #f5efe0);
    border: 1px solid var(--gold-soft);
    border-right: 4px solid var(--gold);
    border-radius: 3px;
    padding: 18px 20px; min-height: 90px;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.04);
}
.arabic-output.empty {
    color: rgba(13,61,58,0.4); font-style: italic; font-size: 16px;
    text-align: center; direction: ltr;
}

/* Style/style probability label panel */
.style-result {
    text-align: center; padding: 12px 0;
}
.style-result .label-en {
    font-family: 'Cormorant Garamond', serif;
    font-size: 14px; letter-spacing: 0.18em;
    color: var(--ink); opacity: 0.7; text-transform: uppercase;
}
.style-result .label-style {
    font-family: 'Cormorant Garamond', serif;
    font-size: 36px; font-weight: 600;
    color: var(--maroon); margin: 4px 0;
}
.style-result .label-conf {
    font-family: 'Inter', sans-serif;
    font-size: 13px; color: var(--ink); opacity: 0.7;
}

/* Section divider */
.divider {
    text-align: center; color: var(--gold);
    font-size: 18px; letter-spacing: 14px;
    margin: 18px 0 6px; user-select: none;
}

/* Footer */
#footer {
    text-align: center; padding: 24px 16px 16px;
    margin-top: 24px;
    border-top: 1px solid var(--gold-soft);
    color: var(--ink); font-size: 12px; opacity: 0.7;
    font-family: 'Cormorant Garamond', serif; font-style: italic;
}

/* Tweak gradio internals */
.gr-box, .form { background: transparent !important; border: none !important; }
label > span { color: var(--ink-dark) !important; font-weight: 500 !important; }
input, textarea, select { font-family: 'Inter', sans-serif !important; }
"""

ARABIC_TITLE = "تَعَرُّفُ الخَطِّ العَرَبِيِّ"

HEADER_HTML = f"""
<div id='title-banner'>
  <div class='ornament'>❋ ❋ ❋</div>
  <div class='arabic-title' dir='rtl'>{ARABIC_TITLE}</div>
  <h1>Arabic Calligraphy Recognition</h1>
  <div class='subtitle'>Style identification &amp; OCR transcription · ECE 693 · AUB</div>
</div>
"""

FOOTER_HTML = """
<div id='footer'>
  Wissam Tedros · Ghina Daoud · Rami Al-Khatib<br/>
  American University of Beirut — MSFEA · ECE 693
</div>
"""

EMPTY_AR_OUTPUT = (
    "<div class='arabic-output empty'>"
    "Upload an image and press <b>Transcribe</b> to begin"
    "</div>"
)


def render_arabic(text: str, latency: float, model_name: str) -> str:
    if not text:
        return EMPTY_AR_OUTPUT
    return (
        f"<div class='arabic-output' dir='rtl' lang='ar'>{text}</div>"
        f"<div style='font-family:Inter;font-size:11px;color:#0d3d3a;"
        f"opacity:0.55;margin-top:6px;text-align:right;direction:ltr'>"
        f"{model_name} · {latency:.2f}s</div>"
    )


def render_style(style: str, conf: float, note: str) -> str:
    if note:
        return (
            f"<div class='style-result'>"
            f"<div class='label-en'>Predicted Style</div>"
            f"<div class='label-style' style='font-size:18px;color:#6e2a3e'>"
            f"checkpoint not loaded</div>"
            f"<div class='label-conf' style='font-style:italic'>{note}</div>"
            f"</div>"
        )
    return (
        f"<div class='style-result'>"
        f"<div class='label-en'>Predicted Style</div>"
        f"<div class='label-style'>{style}</div>"
        f"<div class='label-conf'>confidence {conf*100:.1f}%</div>"
        f"</div>"
    )


# ── Pipeline functions ──────────────────────────────────────────────────────

def predict_style(image: Image.Image | None):
    if image is None:
        return render_style("—", 0.0, "Upload an image first."), {}, "—"
    cls = get_classifier()
    res = cls.predict(image)
    bar = res.probabilities if res.probabilities else {c: 0.0 for c in CLASSES}
    style_label = res.style if not res.note else "—"
    return render_style(res.style, res.confidence, res.note), bar, style_label


def transcribe(
    image: Image.Image | None,
    model_name: str,
    use_style_hint: bool,
    style_label: str,
):
    if image is None:
        return EMPTY_AR_OUTPUT
    ocr = get_ocr()
    style = style_label if (use_style_hint and style_label and style_label != "—") else None
    res = ocr.transcribe(image, model_name=model_name, style_hint=style)
    return render_arabic(res.text, res.latency_s, res.model_name)


def run_full(image, model_name, use_style_hint):
    style_html, prob_dict, style_label = predict_style(image)
    arabic_html = transcribe(image, model_name, use_style_hint, style_label)
    return style_html, prob_dict, style_label, arabic_html


# ── Layout ──────────────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
)

with gr.Blocks(title="Arabic Calligraphy Recognition · ECE 693") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # Left column — input
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<div class='panel-title'>Calligraphy Image</div>")
            input_image = gr.Image(
                type="pil", height=320, sources=["upload", "clipboard"],
                label=None, show_label=False,
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(OCR_MODELS.keys()),
                    value=os.environ.get("OCR_DEFAULT_MODEL", DEFAULT_MODEL),
                    label="OCR model",
                    scale=2,
                )
                use_style = gr.Checkbox(
                    label="Use style hint in prompt",
                    value=True, scale=1,
                )
            run_btn = gr.Button("✦  Transcribe  ✦", variant="primary")

        # Right column — outputs
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                gr.HTML("<div class='panel-title'>Stage 1 · Style Classification</div>")
                style_html = gr.HTML(render_style("—", 0.0, "Awaiting image…"))
                style_label_state = gr.Textbox(visible=False, value="—")
                prob_label = gr.Label(
                    label="Per-class probabilities",
                    num_top_classes=7, show_label=True,
                )

            gr.HTML("<div class='divider'>۞ ۞ ۞</div>")

            with gr.Group(elem_classes="panel"):
                gr.HTML("<div class='panel-title'>Stage 2 · OCR Transcription</div>")
                arabic_html = gr.HTML(EMPTY_AR_OUTPUT)

    gr.HTML(FOOTER_HTML)

    run_btn.click(
        run_full,
        inputs=[input_image, model_dropdown, use_style],
        outputs=[style_html, prob_label, style_label_state, arabic_html],
        api_name="recognize",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("APP_PORT", 7860)),
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=_THEME,
    )
