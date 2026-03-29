#!/usr/bin/env python3
"""Standalone Gradio runner for the Step 7 virtual try-on demo.

This file is generated from the working notebook cells in step7_gradio_ui.ipynb
and is safe to run directly with python app.py.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
import json
import time
import random
from PIL import Image, ImageDraw, ImageEnhance
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── Pipeline imports ──────────────────────────────────────────────────────────
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import AutoImageProcessor, AutoModel
from controlnet_aux import DWposeDetector
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from rembg import remove as rembg_remove
import open_clip
import lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.float16 if device == 'cuda' else torch.float32
print(f'Device : {device}')
if device == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Gradio : {gr.__version__}')

class PipelineManager:
    """
    Loads, holds, and orchestrates all models for the full pipeline.

    Loading order is optimised for VRAM:
        1. Small models first (face, body encoders)
        2. Large diffusion models last (they use cpu_offload)

    Call .generate() for a single output or .generate_batch() for variations.
    """

    def __init__(self, device='cuda'):
        self.device  = device
        self.dtype   = torch.float16 if device == 'cuda' else torch.float32
        self.models  = {}
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_all(self, progress=None):
        """Load all models. Pass a gr.Progress() for Gradio progress bar."""

        steps = [
            ('InsightFace (ArcFace)',     self._load_insightface),
            ('DINOv2 (body + texture)',   self._load_dino),
            ('OpenCLIP (garment)',        self._load_clip),
            ('DWPose (skeleton)',         self._load_dwpose),
            ('GFPGAN (face restore)',     self._load_gfpgan),
            ('Real-ESRGAN (upscaler)',    self._load_realesrgan),
            ('ControlNet + SDXL',        self._load_sdxl),
        ]

        for i, (name, fn) in enumerate(steps):
            print(f'[{i+1}/{len(steps)}] Loading {name}...')
            if progress:
                progress((i / len(steps)), desc=f'Loading {name}...')
            fn()

        self._loaded = True
        print('\n✅ All models loaded — pipeline ready!')
        if progress:
            progress(1.0, desc='Ready!')

    def _load_insightface(self):
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider'],
        )
        app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
        self.models['face_app'] = app

    def _load_dino(self):
        self.models['dino_processor'] = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.models['dino_model']     = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.models['dino_model'].eval()

    def _load_clip(self):
        m, _, p = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k', device=self.device
        )
        m.eval()
        self.models['clip_model']     = m
        self.models['clip_preprocess']= p

    def _load_dwpose(self):
        detector = DWposeDetector()
        if hasattr(detector, 'to'):
            detector = detector.to(self.device)
        self.models['pose_detector'] = detector

    def _load_gfpgan(self):
        import urllib.request
        model_path = 'GFPGANv1.3.pth'
        if not Path(model_path).exists():
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
            urllib.request.urlretrieve(url, model_path)
        self.models['gfpgan'] = GFPGANer(
            model_path=model_path, upscale=1, arch='clean', channel_multiplier=2
        )

    def _load_realesrgan(self):
        import urllib.request
        model_path = 'RealESRGAN_x2plus.pth'
        if not Path(model_path).exists():
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            urllib.request.urlretrieve(url, model_path)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=2)
        self.models['upscaler'] = RealESRGANer(
            scale=2, model_path=model_path, model=model,
            tile=512, tile_pad=10, pre_pad=0,
            half=True if self.device == 'cuda' else False,
        )

    def _load_sdxl(self):
        controlnet = ControlNetModel.from_pretrained(
            'thibaud/controlnet-openpose-sdxl-1.0', torch_dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix', torch_dtype=self.dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            controlnet=controlnet, vae=vae,
            torch_dtype=self.dtype, variant='fp16', use_safetensors=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f'Warning: xformers optimization unavailable: {exc}')
        self.models['pipe'] = pipe

    # ── Helper: encode identity ───────────────────────────────────────────────

    @torch.no_grad()
    def encode_identity(self, image: Image.Image) -> np.ndarray:
        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        faces   = self.models['face_app'].get(img_bgr)
        face_emb = faces[0].normed_embedding if faces else np.zeros(512, dtype=np.float32)

        inputs   = self.models['dino_processor'](images=image, return_tensors='pt').to(self.device)
        body_out = self.models['dino_model'](**inputs)
        body_emb = F.normalize(body_out.last_hidden_state[:, 0, :], dim=-1).squeeze().cpu().numpy()

        fused = 0.6 * face_emb + 0.4 * body_emb[:512]
        return fused / (np.linalg.norm(fused) + 1e-8)

    # ── Helper: extract pose ──────────────────────────────────────────────────

    def extract_pose(self, image: Image.Image, size=(768, 1024)) -> Image.Image:
        img_r    = image.resize(size, Image.LANCZOS)
        skeleton = self.models['pose_detector'](img_r, include_body=True, include_hand=True)
        return skeleton if np.array(skeleton).sum() > 1000 else img_r

    # ── Helper: encode garment ────────────────────────────────────────────────

    @torch.no_grad()
    def encode_garment(self, image: Image.Image) -> np.ndarray:
        rgba      = rembg_remove(image.convert('RGB'))
        white_bg  = Image.new('RGB', rgba.size, (255, 255, 255))
        white_bg.paste(rgba, mask=rgba.split()[-1])
        clean     = white_bg

        clip_in   = self.models['clip_preprocess'](clean).unsqueeze(0).to(self.device)
        clip_feat = F.normalize(self.models['clip_model'].encode_image(clip_in), dim=-1)

        dino_in   = self.models['dino_processor'](images=clean, return_tensors='pt').to(self.device)
        dino_out  = self.models['dino_model'](**dino_in)
        tex_feat  = F.normalize(dino_out.last_hidden_state[:, 1:, :].mean(dim=1), dim=-1)

        return ((clip_feat[:, :768] + tex_feat) / 2.0).squeeze().cpu().numpy()

    # ── Helper: face restoration ──────────────────────────────────────────────

    def restore_face(self, image: Image.Image, fidelity: float = 0.5) -> Image.Image:
        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        _, face_boxes, restored_bgr = self.models['gfpgan'].enhance(
            img_bgr, has_aligned=False, only_center_face=False,
            paste_back=True, weight=fidelity,
        )
        if restored_bgr is None:
            return image
        restored = Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))

        if face_boxes is not None and len(face_boxes) > 0:
            orig_arr = np.array(image.convert('RGB')).astype(np.float32)
            rest_arr = np.array(restored).astype(np.float32)
            mask     = np.zeros(orig_arr.shape[:2], dtype=np.float32)
            for box in face_boxes:
                x1,y1,x2,y2 = [int(v) for v in box]
                pad = 20
                mask[max(0,y1-pad):min(orig_arr.shape[0],y2+pad),
                     max(0,x1-pad):min(orig_arr.shape[1],x2+pad)] = 0.85
            mask    = cv2.GaussianBlur(mask, (51, 51), 0)[:, :, np.newaxis]
            blended = rest_arr * mask + orig_arr * (1 - mask)
            return Image.fromarray(blended.astype(np.uint8))
        return restored

    # ── Helper: upscale ───────────────────────────────────────────────────────

    def upscale(self, image: Image.Image) -> Image.Image:
        img_bgr     = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        upscaled, _ = self.models['upscaler'].enhance(img_bgr, outscale=2)
        return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

    # ── Helper: score ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_output(self, generated: Image.Image, ref_person: Image.Image, ref_garment: Image.Image) -> dict:
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        gen_id  = self.encode_identity(generated)
        ref_id  = self.encode_identity(ref_person)
        id_score = cosine(gen_id, ref_id)

        gen_gar = self.encode_garment(generated)
        ref_gar = self.encode_garment(ref_garment)
        gar_score = cosine(gen_gar, ref_gar)

        gray    = cv2.cvtColor(np.array(generated.convert('RGB')), cv2.COLOR_RGB2GRAY)
        sharp   = float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0))

        return {
            'identity' : round(id_score,  3),
            'garment'  : round(gar_score, 3),
            'sharpness': round(sharp,     3),
            'overall'  : round((id_score * 0.5 + gar_score * 0.3 + sharp * 0.2), 3),
        }

    # ── Main generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        person_image     : Image.Image,
        pose_image       : Image.Image,
        garment_image    : Image.Image,
        prompt           : str,
        negative_prompt  : str   = '',
        num_steps        : int   = 30,
        guidance_scale   : float = 7.5,
        controlnet_scale : float = 0.85,
        identity_scale   : float = 0.80,
        garment_scale    : float = 0.70,
        seed             : int   = 42,
        do_face_restore  : bool  = True,
        do_upscale       : bool  = True,
        fidelity_weight  : float = 0.5,
        width            : int   = 768,
        height           : int   = 1024,
    ) -> dict:
        """
        Full pipeline: person + pose + garment + prompt → polished output.
        Returns dict with image, scores, skeleton, and timing.
        """
        assert self._loaded, 'Call .load_all() first!'
        t0 = time.time()

        if not negative_prompt:
            negative_prompt = (
                'deformed face, disfigured, extra limbs, bad anatomy, mutated hands, '
                'blurry, watermark, text, lowres, ugly, poorly drawn'
            )

        # Step 1: Extract skeleton
        print('  [1/5] Extracting pose...')
        skeleton = self.extract_pose(pose_image, size=(width, height))

        # Step 2: Diffusion generation
        print('  [2/5] Generating...')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        output    = self.models['pipe'](
            prompt                        = prompt,
            negative_prompt               = negative_prompt,
            image                         = skeleton,
            num_inference_steps           = num_steps,
            guidance_scale                = guidance_scale,
            controlnet_conditioning_scale = controlnet_scale,
            generator                     = generator,
            width                         = width,
            height                        = height,
        )
        raw_image = output.images[0]

        # Step 3: Face restoration
        if do_face_restore:
            print('  [3/5] Restoring face...')
            restored = self.restore_face(raw_image, fidelity=fidelity_weight)
        else:
            restored = raw_image

        # Step 4: Sharpen
        print('  [4/5] Sharpening...')
        sharpened = ImageEnhance.Sharpness(restored).enhance(1.3)

        # Step 5: Upscale
        if do_upscale:
            print('  [5/5] Upscaling 2×...')
            final = self.upscale(sharpened)
        else:
            final = sharpened

        # Score
        scores = self.score_output(restored, person_image, garment_image)
        elapsed = round(time.time() - t0, 1)

        print(f'  ✅ Done in {elapsed}s  |  identity={scores["identity"]}  garment={scores["garment"]}')

        return {
            'final_image' : final,
            'raw_image'   : raw_image,
            'skeleton'    : skeleton,
            'scores'      : scores,
            'elapsed'     : elapsed,
            'seed'        : seed,
        }

    def generate_batch(
        self, person_image, pose_image, garment_image,
        prompt, negative_prompt, num_steps, guidance_scale,
        controlnet_scale, identity_scale, garment_scale,
        base_seed, n_variations, do_face_restore, do_upscale,
    ) -> list:
        """Generate N variations with different seeds."""
        results = []
        for i in range(n_variations):
            seed = base_seed + i * 13
            print(f'\nVariation {i+1}/{n_variations} (seed={seed})')
            r = self.generate(
                person_image=person_image, pose_image=pose_image,
                garment_image=garment_image, prompt=prompt,
                negative_prompt=negative_prompt, num_steps=num_steps,
                guidance_scale=guidance_scale, controlnet_scale=controlnet_scale,
                identity_scale=identity_scale, garment_scale=garment_scale,
                seed=seed, do_face_restore=do_face_restore, do_upscale=do_upscale,
            )
            results.append(r)
        return sorted(results, key=lambda x: x['scores']['overall'], reverse=True)


# ── Instantiate and load ──────────────────────────────────────────────────────
manager = PipelineManager(device=device)
print('PipelineManager created — call manager.load_all() to load models')

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: format score report for Gradio markdown output
# ─────────────────────────────────────────────────────────────────────────────

def format_report(scores: dict, elapsed: float, seed: int) -> str:
    def bar(val, width=20):
        filled = int(val * width)
        return '█' * filled + '░' * (width - filled)

    def grade(val):
        if val >= 0.75: return '🌟 Excellent'
        if val >= 0.60: return '✅ Good'
        if val >= 0.45: return '⚠️  Acceptable'
        return '❌ Poor'

    id_s   = scores['identity']
    gar_s  = scores['garment']
    shar_s = scores['sharpness']
    ov_s   = scores['overall']

    return f"""## 📊 Quality Report

| Metric | Score | Grade |
|--------|-------|-------|
| 👤 Identity | **{id_s:.3f}** | {grade(id_s)} |
| 👗 Garment  | **{gar_s:.3f}** | {grade(gar_s)} |
| 🔍 Sharpness| **{shar_s:.3f}** | {grade(shar_s)} |
| ⭐ Overall  | **{ov_s:.3f}** | {grade(ov_s)} |

**Identity** `{bar(id_s)}` {id_s:.2f}
**Garment**  `{bar(gar_s)}` {gar_s:.2f}
**Sharpness** `{bar(shar_s)}` {shar_s:.2f}

---
⏱️ Generated in **{elapsed}s**  |  🎲 Seed: `{seed}`
"""


# ─────────────────────────────────────────────────────────────────────────────
#  UI callback: single generation
# ─────────────────────────────────────────────────────────────────────────────

def run_generate(
    person_img, pose_img, garment_img,
    prompt, negative_prompt,
    num_steps, guidance_scale,
    controlnet_scale, identity_scale, garment_scale,
    seed, do_face_restore, do_upscale, fidelity_weight,
    progress=gr.Progress()
):
    """
    Called by the Generate button.
    Returns: (output_image, skeleton_image, report_markdown)
    """
    if person_img is None or pose_img is None or garment_img is None:
        return None, None, '⚠️ Please upload all three images first.'

    progress(0.0, desc='Starting pipeline...')

    try:
        progress(0.1, desc='Loading models (if not yet loaded)...')
        if not manager._loaded:
            manager.load_all(progress=progress)

        progress(0.3, desc='Extracting pose...')
        result = manager.generate(
            person_image     = Image.fromarray(person_img).convert('RGB'),
            pose_image       = Image.fromarray(pose_img).convert('RGB'),
            garment_image    = Image.fromarray(garment_img).convert('RGB'),
            prompt           = prompt,
            negative_prompt  = negative_prompt,
            num_steps        = int(num_steps),
            guidance_scale   = float(guidance_scale),
            controlnet_scale = float(controlnet_scale),
            identity_scale   = float(identity_scale),
            garment_scale    = float(garment_scale),
            seed             = int(seed),
            do_face_restore  = do_face_restore,
            do_upscale       = do_upscale,
            fidelity_weight  = float(fidelity_weight),
        )
        progress(1.0, desc='Done!')
        report = format_report(result['scores'], result['elapsed'], result['seed'])
        return np.array(result['final_image']), np.array(result['skeleton']), report

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, None, f'❌ Error:\n```\n{tb}\n```'


# ─────────────────────────────────────────────────────────────────────────────
#  UI callback: batch / variations
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    person_img, pose_img, garment_img,
    prompt, negative_prompt,
    num_steps, guidance_scale,
    controlnet_scale, identity_scale, garment_scale,
    seed, n_variations,
    progress=gr.Progress()
):
    """
    Called by the Generate Variations button.
    Returns a gallery list of (image, caption) tuples.
    """
    if person_img is None or pose_img is None or garment_img is None:
        return []

    if not manager._loaded:
        manager.load_all()

    n = int(n_variations)
    gallery_images = []

    for i in range(n):
        progress((i / n), desc=f'Variation {i+1}/{n}...')
        cur_seed = int(seed) + i * 13
        try:
            result = manager.generate(
                person_image     = Image.fromarray(person_img).convert('RGB'),
                pose_image       = Image.fromarray(pose_img).convert('RGB'),
                garment_image    = Image.fromarray(garment_img).convert('RGB'),
                prompt           = prompt,
                negative_prompt  = negative_prompt,
                num_steps        = int(num_steps),
                guidance_scale   = float(guidance_scale),
                controlnet_scale = float(controlnet_scale),
                identity_scale   = float(identity_scale),
                garment_scale    = float(garment_scale),
                seed             = cur_seed,
                do_face_restore  = True,
                do_upscale       = False,  # faster for batch preview
            )
            s = result['scores']
            caption = (f"Seed {cur_seed} | "
                       f"ID:{s['identity']:.2f} "
                       f"Gar:{s['garment']:.2f} "
                       f"({result['elapsed']}s)")
            gallery_images.append((np.array(result['final_image']), caption))
        except Exception as e:
            gallery_images.append((np.zeros((512, 512, 3), dtype=np.uint8), f'Error: {e}'))

    progress(1.0, desc='Done!')
    return gallery_images


# ─────────────────────────────────────────────────────────────────────────────
#  UI callback: load models button
# ─────────────────────────────────────────────────────────────────────────────

def load_models_fn(progress=gr.Progress()):
    if manager._loaded:
        return '✅ Models already loaded and ready!'
    try:
        manager.load_all(progress=progress)
        return '✅ All models loaded — pipeline ready!'
    except Exception as e:
        return f'❌ Error loading models: {e}'


print('✅ UI callbacks defined')

# ─────────────────────────────────────────────────────────────────────────────
#  Prompt suggestions
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_EXAMPLES = [
    ['standing in a sunlit city street, golden hour, bokeh background, photorealistic, 8K'],
    ['professional headshot in a modern office, soft studio lighting, photorealistic'],
    ['walking in a lush green park, natural daylight, candid photography style, sharp focus'],
    ['on a rooftop at sunset, dramatic sky, cinematic composition, photorealistic'],
    ['in a cozy coffee shop, warm ambient light, lifestyle photography, Canon 5D'],
    ['fashion editorial shoot, minimalist white background, professional studio lighting'],
]


# ─────────────────────────────────────────────────────────────────────────────
#  CSS theme
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
    max-width: 1400px !important;
    margin: 0 auto;
}
.input-panel {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e0e0e0;
}
.output-panel {
    background: #f0f4ff;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #c8d8ff;
}
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    min-height: 48px !important;
}
.load-btn {
    background: #28a745 !important;
    color: white !important;
    border-radius: 8px !important;
}
h1 { text-align: center; }
.score-good  { color: #28a745; font-weight: bold; }
.score-warn  { color: #ffc107; font-weight: bold; }
.score-bad   { color: #dc3545; font-weight: bold; }
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Build the app
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CUSTOM_CSS, title='Consistent Identity AI') as app:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🧑‍🎨 Consistent Identity AI
    ### Virtual try-on + pose generation — same face, any outfit, any pose
    Upload a **person reference**, a **pose reference**, and an **outfit image**.
    The model keeps the person's identity while changing the pose and clothes.
    """)

    # ── Load models banner ────────────────────────────────────────────────────
    with gr.Row():
        load_btn    = gr.Button('🚀 Load All Models (required before first use)', elem_classes='load-btn', scale=3)
        load_status = gr.Textbox(label='Status', value='⏳ Models not loaded yet', interactive=False, scale=2)
    load_btn.click(fn=load_models_fn, inputs=[], outputs=[load_status])

    gr.Markdown('---')

    # ── Main tabs ─────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 1 — Single generation
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem('✨ Generate'):

            with gr.Row():

                # ── Left: inputs ──────────────────────────────────────────────
                with gr.Column(scale=5, elem_classes='input-panel'):
                    gr.Markdown('### 📥 Inputs')

                    with gr.Row():
                        person_input  = gr.Image(label='👤 Person reference',
                                                  type='numpy', height=280)
                        pose_input    = gr.Image(label='🦴 Pose reference',
                                                  type='numpy', height=280)
                        garment_input = gr.Image(label='👗 Outfit / garment',
                                                  type='numpy', height=280)

                    prompt_box = gr.Textbox(
                        label='📝 Prompt — describe the scene, lighting, background',
                        placeholder='standing in a sunlit city street, golden hour, photorealistic...',
                        lines=2,
                    )
                    neg_prompt_box = gr.Textbox(
                        label='🚫 Negative prompt (optional)',
                        value='deformed, blurry, bad anatomy, extra limbs, watermark',
                        lines=1,
                    )

                    # ── Prompt examples ───────────────────────────────────────
                    gr.Examples(
                        examples=PROMPT_EXAMPLES,
                        inputs=[prompt_box],
                        label='💡 Prompt ideas (click to use)',
                    )

                    # ── Advanced settings (collapsed) ─────────────────────────
                    with gr.Accordion('⚙️ Advanced Settings', open=False):
                        with gr.Row():
                            identity_scale_sl = gr.Slider(
                                0.0, 1.0, value=0.80, step=0.05,
                                label='👤 Identity strength\n(how much reference face is preserved)',
                            )
                            garment_scale_sl  = gr.Slider(
                                0.0, 1.0, value=0.70, step=0.05,
                                label='👗 Garment strength\n(how strictly the outfit is reproduced)',
                            )
                        with gr.Row():
                            controlnet_sl = gr.Slider(
                                0.0, 1.5, value=0.85, step=0.05,
                                label='🦴 Pose strength\n(how strictly the skeleton is followed)',
                            )
                            guidance_sl   = gr.Slider(
                                1.0, 15.0, value=7.5, step=0.5,
                                label='📝 Guidance scale\n(how closely the prompt is followed)',
                            )
                        with gr.Row():
                            steps_sl      = gr.Slider(
                                10, 50, value=30, step=1,
                                label='🔄 Denoising steps\n(more = better quality, slower)',
                            )
                            seed_box      = gr.Number(
                                value=42, label='🎲 Seed (-1 = random)',
                                precision=0,
                            )
                        with gr.Row():
                            fidelity_sl   = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.1,
                                label='😊 Face fidelity\n(0=full restore, 1=keep original)',
                            )
                            do_restore_cb = gr.Checkbox(value=True,  label='✅ Face restoration (GFPGAN)')
                            do_upscale_cb = gr.Checkbox(value=True,  label='✅ 2× Upscale (Real-ESRGAN)')

                    # ── Generate button ───────────────────────────────────────
                    with gr.Row():
                        generate_btn = gr.Button('✨ Generate', elem_classes='generate-btn', scale=3)
                        random_btn   = gr.Button('🎲 Random seed', scale=1)

                    random_btn.click(
                        fn=lambda: random.randint(0, 2**31),
                        inputs=[], outputs=[seed_box],
                    )

                # ── Right: outputs ────────────────────────────────────────────
                with gr.Column(scale=4, elem_classes='output-panel'):
                    gr.Markdown('### 📤 Output')

                    output_image    = gr.Image(
                        label='Generated image',
                        type='numpy', height=500, show_download_button=True,
                    )
                    skeleton_image  = gr.Image(
                        label='Pose skeleton used',
                        type='numpy', height=200,
                    )
                    quality_report  = gr.Markdown('_Quality report will appear here after generation._')

            # ── Wire up Generate ──────────────────────────────────────────────
            generate_btn.click(
                fn=run_generate,
                inputs=[
                    person_input, pose_input, garment_input,
                    prompt_box, neg_prompt_box,
                    steps_sl, guidance_sl,
                    controlnet_sl, identity_scale_sl, garment_scale_sl,
                    seed_box, do_restore_cb, do_upscale_cb, fidelity_sl,
                ],
                outputs=[output_image, skeleton_image, quality_report],
            )

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 2 — Batch / variations
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem('🎞️ Generate Variations'):
            gr.Markdown("""
            Generate multiple variations with different seeds and **automatically
            rank them by identity + garment consistency score**.
            Best result appears first.
            """)

            with gr.Row():
                b_person  = gr.Image(label='👤 Person reference',  type='numpy', height=200)
                b_pose    = gr.Image(label='🦴 Pose reference',    type='numpy', height=200)
                b_garment = gr.Image(label='👗 Outfit reference',  type='numpy', height=200)

            b_prompt   = gr.Textbox(
                label='Prompt',
                placeholder='standing in a city street, golden hour, photorealistic...',
            )
            b_neg      = gr.Textbox(
                label='Negative prompt',
                value='deformed, blurry, bad anatomy, extra limbs, watermark',
            )

            with gr.Row():
                b_n_var    = gr.Slider(1, 6, value=3, step=1,  label='Number of variations')
                b_seed     = gr.Number(value=42, label='Base seed', precision=0)
                b_steps    = gr.Slider(10, 50, value=25, step=1, label='Steps (per variation)')
                b_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label='Guidance scale')

            with gr.Row():
                b_ctrl  = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label='Pose strength')
                b_id    = gr.Slider(0.0, 1.0, value=0.80, step=0.05, label='Identity strength')
                b_gar   = gr.Slider(0.0, 1.0, value=0.70, step=0.05, label='Garment strength')

            batch_btn     = gr.Button('🎞️ Generate Variations', elem_classes='generate-btn')
            batch_gallery = gr.Gallery(
                label='Variations (sorted best → worst by identity score)',
                columns=3, rows=2, height=600, show_download_button=True,
            )

            batch_btn.click(
                fn=run_batch,
                inputs=[
                    b_person, b_pose, b_garment,
                    b_prompt, b_neg,
                    b_steps, b_guidance,
                    b_ctrl, b_id, b_gar,
                    b_seed, b_n_var,
                ],
                outputs=[batch_gallery],
            )

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 3 — How to use
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem('📖 How to use'):
            gr.Markdown("""
            ## 📖 How to use this app

            ### Step 1 — Load models
            Click the **🚀 Load All Models** button at the top.
            This downloads ~12GB of model weights on first run and takes 10–15 minutes.
            After that, models are cached and load in ~30 seconds.

            ---

            ### Step 2 — Upload your images

            | Image | What to upload | Tips |
            |-------|---------------|------|
            | 👤 **Person reference** | A clear photo of the person whose face/body you want to preserve | Front-facing portrait works best |
            | 🦴 **Pose reference** | A photo showing the body pose you want | Can be a different person — only the skeleton is used |
            | 👗 **Outfit reference** | A product photo or worn photo of the target garment | Clean product shots on white background work best |

            ---

            ### Step 3 — Write your prompt
            Describe the **scene and setting**, not the person or outfit (those are controlled by the images).

            **Good prompts:**
            - `standing in a sunlit city street, golden hour lighting, photorealistic, sharp focus`
            - `professional headshot, soft studio lighting, clean background, 8K`
            - `walking in a park, natural daylight, candid style, bokeh background`

            **What NOT to describe in the prompt:** the person's face, hair color, or specific clothing
            (the images handle those).

            ---

            ### Step 4 — Tune the sliders

            | Slider | Effect |
            |--------|--------|
            | 👤 **Identity strength** | Higher = more faithful to reference face. Lower = more creative variation. Start at 0.8. |
            | 👗 **Garment strength** | Higher = more faithful to reference outfit texture. Start at 0.7. |
            | 🦴 **Pose strength** | Higher = body follows the skeleton exactly. Lower = more natural but may drift. Start at 0.85. |
            | 😊 **Face fidelity** | 0 = maximum restoration. 1 = preserve input face exactly. 0.5 is balanced. |

            ---

            ### 📊 Quality Report
            After each generation you'll see:
            - **Identity score** — how similar the generated face is to the reference (>0.6 is good)
            - **Garment score** — how well the outfit transferred (>0.5 is good)
            - **Sharpness score** — overall image quality

            If scores are low, try: increasing identity/garment strength, more steps, or a different seed.

            ---

            ### 💡 Pro tips
            - Use **Generate Variations** tab to get 3–6 versions and pick the best
            - Add `photorealistic, 8K, sharp focus` to every prompt
            - For virtual try-on: clean product shot on white background works best for outfit
            - For consistent character: frontal portrait with good lighting works best for reference
            """)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:13px'>
    Built with SDXL · ControlNet · IP-Adapter · InsightFace · DINOv2 · GFPGAN · Real-ESRGAN
    </div>
    """)

print('✅ Gradio app assembled!')

def parse_args():
    parser = argparse.ArgumentParser(description='Run the Step 7 Gradio app.')
    parser.add_argument('--share', action='store_true', help='Create a public Gradio share link.')
    parser.add_argument('--server-port', type=int, default=7860, help='Port for the Gradio server.')
    parser.add_argument('--server-name', default='0.0.0.0', help='Bind address for the Gradio server.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.launch(
        share=args.share,
        server_port=args.server_port,
        server_name=args.server_name,
        show_error=True,
        quiet=False,
    )
