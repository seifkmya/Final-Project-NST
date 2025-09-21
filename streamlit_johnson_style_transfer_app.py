# Streamlit Neural Style Transfer Suite
# -------------------------------------------------
# Tabs:
# 1) Johnson (Per-style, fast) — load a pre-trained TransformNet (.pth) and stylize images
# 2) AdaIN (Arbitrary style, decoder) — upload content + style, load decoder (.pth), and stylize with alpha
#
# Features
# - Auto device (CUDA if available)
# - Batch stylization for Johnson; single-pair stylization for AdaIN
# - Keep-original or resize options (Johnson: max long side; AdaIN: max short side)
# - Side-by-side preview(s), progress bars, and downloads (single or zip)
# - Safe state-dict loading (handles DataParallel 'module.' prefix)
# -------------------------------------------------

import io
import os
import time
import hashlib
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import torchvision.transforms as T
from torchvision import models

# --------------------- Shared helpers ---------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

@st.cache_data(show_spinner=False)
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# Basic tensor-PIL conversions (expects [0,1])
def pil_to_tensor_unit(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def tensor_to_pil_unit(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(t)

# Resize helpers
def resize_keep_ar_long(img: Image.Image, max_long_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_long_side:
        return img
    if w >= h:
        new_w = max_long_side
        new_h = int(round(h * (max_long_side / w)))
    else:
        new_h = max_long_side
        new_w = int(round(w * (max_long_side / h)))
    return img.resize((new_w, new_h), Image.BICUBIC)

def resize_keep_ar_short(img: Image.Image, max_short_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) >= max_short_side:
        return img
    if w <= h:
        new_w = max_short_side
        new_h = int(round(h * (max_short_side / w)))
    else:
        new_h = max_short_side
        new_w = int(round(w * (max_short_side / h)))
    return img.resize((new_w, new_h), Image.BICUBIC)

# --------------------- Johnson (Per-style) ---------------------
class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        pad = kernel // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride)
    def forward(self, x):
        return self.conv(self.pad(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1   = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2   = nn.InstanceNorm2d(channels, affine=True)
    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return x + y

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, upsample=None):
        super().__init__()
        self.upsample = upsample
        self.pad = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(in_c, out_c, kernel, 1)
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.conv(self.pad(x))

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1   = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2   = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3   = nn.InstanceNorm2d(128, affine=True)
        # Residuals
        self.res   = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # Decoder
        self.up1   = UpsampleConvLayer(128, 64, 3, upsample=2)
        self.in4   = nn.InstanceNorm2d(64, affine=True)
        self.up2   = UpsampleConvLayer(64, 32, 3, upsample=2)
        self.in5   = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = ConvLayer(32, 3, 9, 1)
    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        y = self.res(y)
        y = F.relu(self.in4(self.up1(y)))
        y = F.relu(self.in5(self.up2(y)))
        y = torch.tanh(self.conv4(y))
        return (y + 1) / 2.0  # [0,1]

@st.cache_resource(show_spinner=False)
def load_johnson_model(model_bytes: Optional[bytes], model_path: Optional[str], device_str: str):
    device = torch.device(device_str)
    net = TransformNet().to(device)
    if model_bytes is not None:
        ckpt = torch.load(io.BytesIO(model_bytes), map_location=device)
    elif model_path is not None and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("No valid Johnson .pth provided. Upload a file or enter a correct path.")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = ckpt
    cleaned = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        st.warning(f"Johnson loaded with non-strict keys. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net

# --------------------- AdaIN (Arbitrary) ---------------------
LAYER_NAME_MAP = { 1:'relu1_1', 6:'relu2_1', 11:'relu3_1', 20:'relu4_1' }
STYLE_LAYERS = ['relu1_1','relu2_1','relu3_1','relu4_1']
CONTENT_LAYER = 'relu4_1'

class VGGEncoder(nn.Module):
    '''VGG-19 up to relu4_1 (frozen). Returns dict of requested layers.'''
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
    def forward(self, x, out_keys=('relu4_1',)):
        feats = {}
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            name = LAYER_NAME_MAP.get(i, None)
            if name in out_keys:
                feats[name] = h
            if i >= 20:  # up to relu4_1
                pass
        return feats

class AdaINDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3)
        )
    def forward(self, x):
        return self.body(x)

@st.cache_resource(show_spinner=False)
def load_adain_models(decoder_bytes: Optional[bytes], decoder_path: Optional[str], device_str: str):
    device = torch.device(device_str)
    enc = VGGEncoder().to(device).eval()
    dec = AdaINDecoder().to(device)
    # Load checkpoint
    if decoder_bytes is not None:
        ckpt = torch.load(io.BytesIO(decoder_bytes), map_location=device)
    elif decoder_path is not None and Path(decoder_path).exists():
        ckpt = torch.load(decoder_path, map_location=device)
    else:
        raise FileNotFoundError("No valid AdaIN decoder .pth provided. Upload a file or enter a correct path.")
    if isinstance(ckpt, dict) and 'decoder' in ckpt:
        sd = ckpt['decoder']
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = ckpt
    dec.load_state_dict(sd, strict=True)
    dec.eval()
    for p in dec.parameters(): p.requires_grad_(False)
    return enc, dec

# AdaIN core
def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    B, C = feat.size()[:2]
    var = feat.view(B, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(B, C, 1, 1)
    mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    return mean, std

def adain_op(content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-5):
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

# VGG preprocessing / postprocessing
def to_vgg_tensor(img: Image.Image, device: torch.device, max_short_side: int) -> torch.Tensor:
    img = ImageOps.exif_transpose(img.convert('RGB'))
    img = resize_keep_ar_short(img, max_short_side)
    t = T.ToTensor()(img)
    t = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(t)
    return t.unsqueeze(0).to(device)

@torch.no_grad()
def adain_stylize_pil(content_img: Image.Image, style_img: Image.Image, enc: VGGEncoder, dec: AdaINDecoder, device: torch.device, alpha: float = 1.0, max_short_side: int = 512) -> Image.Image:
    # Encode
    c = to_vgg_tensor(content_img, device, max_short_side)
    s = to_vgg_tensor(style_img, device, max_short_side)
    c4 = enc(c, out_keys=['relu4_1'])['relu4_1']
    s4 = enc(s, out_keys=['relu4_1'])['relu4_1']
    t = adain_op(c4, s4)
    if alpha < 1.0:
        t = alpha * t + (1 - alpha) * c4
    y = dec(t).clamp(-3, 3)
    # Denormalize back to [0,1]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1,3,1,1)
    y = torch.clamp(y * std + mean, 0, 1)
    return tensor_to_pil_unit(y)

# --------------------- UI ---------------------
st.set_page_config(page_title="Neural Style Transfer Suite", page_icon="🎨", layout="wide")

st.markdown('<style>\n.small-muted { color: var(--text-color-secondary); font-size: 0.9rem; }\n.imgbox { border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; padding: 6px; }\n</style>', unsafe_allow_html=True)

st.title("🎨 Neural Style Transfer Suite")
st.caption("Johnson (fast, per-style) and AdaIN (arbitrary style) in one app.")

# Sidebar (shared)
with st.sidebar:
    st.header("Runtime & Output")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_fmt = st.selectbox("Output format", ["PNG", "JPEG"], index=0)
    jpeg_q = st.slider("JPEG quality", 70, 100, 95) if out_fmt == "JPEG" else None
    st.markdown("---")
    st.markdown("**Tip**: If you hit memory limits, resize images smaller.")

# Tabs
_tab1, _tab2 = st.tabs(["Johnson (Per-style)", "AdaIN (Arbitrary)"])

# --------------------- Johnson Tab ---------------------
with _tab1:
    st.subheader("Johnson TransformNet — Inference")
    st.markdown("Upload or point to a pre-trained Johnson model (.pth). Then upload one or more content images.")

    colA, colB = st.columns(2)
    with colA:
        st.write("**Model (.pth)**")
        j_up = st.file_uploader("Upload Johnson checkpoint", type=["pth", "pt"], accept_multiple_files=False, key="j_ckpt")
        j_model_path = st.text_input("...or path on server", value="transform_net.pth", key="j_path")
        keep_size = st.checkbox("Keep original resolution", value=True, key="j_keep")
        max_long = st.slider("Max long side (if resizing)", 256, 2048, 1024, step=64, key="j_maxlong")
    with colB:
        st.write("**Content image(s)**")
        j_files = st.file_uploader("Images to stylize", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True, key="j_imgs")

    # Resolve model
    j_bytes = j_up.read() if j_up is not None else None
    try:
        j_net = load_johnson_model(j_bytes, j_model_path if j_bytes is None else None, device_str)
        j_device = next(j_net.parameters()).device
    except Exception as e:
        st.error(f"Johnson model load error: {e}")
        j_net = None
        j_device = torch.device(device_str)

    # Preview & run
    results_j: List[Tuple[str, Image.Image]] = []
    if j_files:
        cols = st.columns(2)
        with cols[0]:
            st.write("**Original (first)**")
            j_raw = Image.open(j_files[0]).convert("RGB")
            j_raw = ImageOps.exif_transpose(j_raw)
            st.image(j_raw, use_column_width=True, output_format='PNG', clamp=True)

        if st.button("✨ Stylize (Johnson)", type='primary', use_container_width=True, key="j_run") and j_net is not None:
            t0 = time.time()
            bar = st.progress(0, text="Stylizing...")
            for i, f in enumerate(j_files, start=1):
                img = Image.open(f).convert("RGB")
                img = ImageOps.exif_transpose(img)
                if not keep_size:
                    img = resize_keep_ar_long(img, max_long)
                x = pil_to_tensor_unit(img, j_device)
                with torch.no_grad():
                    y = j_net(x)
                out_pil = tensor_to_pil_unit(y)
                results_j.append((f.name, out_pil))
                bar.progress(i/len(j_files), text=f"Stylized {i}/{len(j_files)}")
            bar.empty()
            st.success(f"Johnson done in {time.time()-t0:.2f}s for {len(j_files)} image(s).")

    if results_j:
        st.markdown("---")
        st.subheader("Download — Johnson")
        if len(results_j) == 1:
            name, im = results_j[0]
            b = io.BytesIO()
            save_name = Path(name).stem + (".png" if out_fmt=="PNG" else ".jpg")
            if out_fmt == "PNG": im.save(b, format='PNG')
            else: im.save(b, format='JPEG', quality=jpeg_q, optimize=True)
            st.download_button(f"Download {save_name}", b.getvalue(), file_name=save_name, mime='image/png' if out_fmt=="PNG" else 'image/jpeg', use_container_width=True)
        else:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, im in results_j:
                    out_name = Path(name).stem + (".png" if out_fmt=="PNG" else ".jpg")
                    bb = io.BytesIO()
                    if out_fmt == "PNG": im.save(bb, format='PNG')
                    else: im.save(bb, format='JPEG', quality=jpeg_q, optimize=True)
                    zf.writestr(out_name, bb.getvalue())
            st.download_button(f"Download {len(results_j)} images (zip)", zip_buf.getvalue(), file_name="stylized_batch_johnson.zip", mime="application/zip", use_container_width=True)

# --------------------- AdaIN Tab ---------------------
with _tab2:
    st.subheader("AdaIN — Arbitrary Style Inference")
    st.markdown("Upload **decoder_final.pth** (or a checkpoint containing 'decoder') plus one content and one style image.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Decoder checkpoint (.pth)**")
        a_up = st.file_uploader("Upload AdaIN decoder", type=["pth", "pt"], accept_multiple_files=False, key="a_ckpt")
        a_model_path = st.text_input("...or path on server", value="decoder_final.pth", key="a_path")
        alpha = st.slider("Style strength (alpha)", 0.0, 1.0, 1.0, 0.05, key="a_alpha")
        max_short = st.slider("Resize **shorter** side to", 256, 1024, 512, step=32, key="a_short")
    with c2:
        st.write("**Content & Style**")
        a_content = st.file_uploader("Content image", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=False, key="a_content")
        a_style   = st.file_uploader("Style image",   type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=False, key="a_style")

    # Load models
    a_bytes = a_up.read() if a_up is not None else None
    try:
        a_enc, a_dec = load_adain_models(a_bytes, a_model_path if a_bytes is None else None, device_str)
        a_device = next(a_dec.parameters()).device
    except Exception as e:
        st.error(f"AdaIN model load error: {e}")
        a_enc = None; a_dec = None; a_device = torch.device(device_str)

    # Preview inputs
    if a_content and a_style:
        cc, cs = st.columns(2)
        with cc:
            st.write("**Content (preview)**")
            img_c = Image.open(a_content).convert('RGB')
            st.image(ImageOps.exif_transpose(img_c), use_column_width=True)
        with cs:
            st.write("**Style (preview)**")
            img_s = Image.open(a_style).convert('RGB')
            st.image(ImageOps.exif_transpose(img_s), use_column_width=True)

    # Run AdaIN
    result_a: Optional[Image.Image] = None
    if st.button("✨ Stylize (AdaIN)", type='primary', use_container_width=True, key="a_run") and a_enc is not None and a_dec is not None and a_content and a_style:
        img_c = Image.open(a_content).convert('RGB')
        img_s = Image.open(a_style).convert('RGB')
        with st.spinner("Running AdaIN..."):
            result_a = adain_stylize_pil(img_c, img_s, a_enc, a_dec, a_device, alpha=alpha, max_short_side=max_short)
        st.success("AdaIN stylization complete.")

    if result_a is not None:
        st.markdown("---")
        st.subheader("Result — AdaIN")
        st.image(result_a, use_column_width=True)
        # Download
        buf = io.BytesIO()
        save_name = "adain_stylized.png" if out_fmt=="PNG" else "adain_stylized.jpg"
        if out_fmt == "PNG": result_a.save(buf, format='PNG')
        else: result_a.save(buf, format='JPEG', quality=jpeg_q, optimize=True)
        st.download_button(f"Download {save_name}", buf.getvalue(), file_name=save_name, mime='image/png' if out_fmt=="PNG" else 'image/jpeg', use_container_width=True)

# Footer
st.markdown("---")
st.caption("Tip: Johnson is best for a single trained style at speed; AdaIN is best when you want any style with one decoder.")
