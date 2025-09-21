# Streamlit Johnson Style Transfer UI
# -------------------------------------------------
# Features
# - Upload or point to a Johnson-style .pth model and stylize single or multiple images
# - Auto device (CUDA if available)
# - Keep original resolution or auto-resize by max long side
# - Side-by-side preview, progress bars, zip download for batch
# - Safe state-dict loading (handles DataParallel 'module.' prefix)
# -------------------------------------------------

import io
import os
import time
import hashlib
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# --------------------- Transform Net (Johnson et al.) ---------------------
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
        return (y + 1) / 2.0  # map from [-1,1] to [0,1]

# --------------------- Helpers ---------------------
@st.cache_data(show_spinner=False)
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

@st.cache_resource(show_spinner=False)
def load_model(model_bytes: Optional[bytes], model_path: Optional[str], device_str: str):
    """Load and cache the Johnson transform net. Accept bytes (uploaded) or a local path.
    Returns a model on the requested device in eval() mode.
    """
    device = torch.device(device_str)
    net = TransformNet().to(device)

    # Resolve checkpoint
    if model_bytes is not None:
        ckpt = torch.load(io.BytesIO(model_bytes), map_location=device)
    elif model_path is not None and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("No valid Johnson .pth provided. Upload a file or enter a correct path.")

    # Some checkpoints save {'state_dict': ...} or have DataParallel 'module.' prefix
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        # raw state_dict tensor map
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        nk = k.replace('module.', '')
        cleaned[nk] = v

    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        st.warning(f"Loaded with non-strict keys. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net

def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    # Keep dynamic range [0,1]
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(t)

def resize_keep_ar(img: Image.Image, max_long_side: int) -> Image.Image:
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

def stylize_pil(img: Image.Image, net: nn.Module, device: torch.device) -> Image.Image:
    with torch.no_grad():
        x = pil_to_tensor(img, device)
        y = net(x)
        return tensor_to_pil(y.float())

# --------------------- UI ---------------------
st.set_page_config(
    page_title="Johnson Style Transfer UI",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .small-muted { color: var(--text-color-secondary); font-size: 0.9rem; }
    .imgbox { border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; padding: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎨 Johnson Neural Style Transfer (Inference UI)")
st.caption("Load a pre-trained Johnson transform network (.pth) and stylize images with ease.")

# Sidebar controls
with st.sidebar:
    st.header("Model & Runtime")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    st.subheader("Johnson .pth")
    up = st.file_uploader("Upload Johnson checkpoint (.pth/.pt)", type=["pth", "pt"], accept_multiple_files=False)
    model_path_text = st.text_input("...or path on server", value="transform_net.pth")

    keep_size = st.checkbox("Keep original resolution", value=True)
    max_side = st.slider("Max long side (if resizing)", 256, 2048, 1024, step=64)

    out_fmt = st.selectbox("Output format", ["PNG", "JPEG"], index=0)
    jpeg_q =  st.slider("JPEG quality", 70, 100, 95) if out_fmt == "JPEG" else None

    st.markdown("---")
    st.markdown("**Tip**: For large photos, disable 'Keep original' or lower 'Max long side' if you hit memory limits.")

# Resolve model once per run
model_bytes = up.read() if up is not None else None

# Use cached loader
try:
    net = load_model(model_bytes, model_path_text if model_bytes is None else None, device_str)
    device = next(net.parameters()).device
except Exception as e:
    st.error(f"Model load error: {e}")
    net = None
    device = torch.device(device_str)

# Content images
st.subheader("1) Upload content image(s)")
files = st.file_uploader("Images to stylize", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)

col_l, col_r = st.columns(2)

results = []
if files:
    # Preview first image
    with col_l:
        st.write("**Original (first)**")
        raw = Image.open(files[0]).convert("RGB")
        raw = ImageOps.exif_transpose(raw)
        st.image(raw, use_column_width=True, output_format='PNG', clamp=True)

    run_btn = st.button("✨ Stylize", type='primary', use_container_width=True)

    if run_btn and net is not None:
        t0 = time.time()
        bar = st.progress(0, text="Stylizing...")
        for i, f in enumerate(files, start=1):
            img = Image.open(f).convert("RGB")
            img = ImageOps.exif_transpose(img)
            if not keep_size:
                img = resize_keep_ar(img, max_side)
            out = stylize_pil(img, net, device)
            results.append((f.name, out))
            bar.progress(i/len(files), text=f"Stylized {i}/{len(files)}")
        bar.empty()
        st.success(f"Done in {time.time()-t0:.2f}s for {len(files)} image(s).")

    if results:
        # Show first result side-by-side
        with col_r:
            st.write("**Stylized (first)**")
            st.image(results[0][1], use_column_width=True, output_format='PNG', clamp=True)

        st.markdown("---")
        st.subheader("2) Download")
        # Single or zip
        if len(results) == 1:
            name, im = results[0]
            buf = io.BytesIO()
            save_name = Path(name).stem + (".png" if out_fmt=="PNG" else ".jpg")
            if out_fmt == "PNG":
                im.save(buf, format='PNG')
            else:
                im.save(buf, format='JPEG', quality=jpeg_q, optimize=True)
            st.download_button(
                label=f"Download {save_name}",
                data=buf.getvalue(),
                file_name=save_name,
                mime='image/png' if out_fmt=="PNG" else 'image/jpeg',
                use_container_width=True,
            )
        else:
            # zip many
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, im in results:
                    out_name = Path(name).stem + (".png" if out_fmt=="PNG" else ".jpg")
                    b = io.BytesIO()
                    if out_fmt == "PNG":
                        im.save(b, format='PNG')
                    else:
                        im.save(b, format='JPEG', quality=jpeg_q, optimize=True)
                    zf.writestr(out_name, b.getvalue())
            st.download_button(
                label=f"Download {len(results)} images (zip)",
                data=zip_buf.getvalue(),
                file_name="stylized_batch.zip",
                mime="application/zip",
                use_container_width=True,
            )
else:
    st.info("Upload one or more content images to get started.")

# Footer
st.markdown("---")
st.caption("Next: we can add an AdaIN tab and optional per-style training controls. 💡")
