# DPU + camera overlay for the Kria-PYNQ Jupyter notebook on the KV260.
# Loads the design with PYNQ only (no xmutil) and runs one DPU inference.
#
# One-time setup:
#   1. On the build host: impl/scripts/copy_pynq_overlay.sh
#      Copy build/pynq/ to OVERLAY_DIR on the KV260 (kv260-camdpu.bit,
#      .hwh, .xclbin, .dtbo).
#   2. On the KV260, after each restart: sudo xmutil unloadapp
#      PYNQ and xmutil must not both load a design.
#
# Why XLNX_VART_FIRMWARE is set: load_model() always points /etc/vart.conf at
# /usr/lib/dpu.xclbin, the stock PYNQ DPU. That xclbin describes another DPU
# design, and VART can hang on it. VART reads XLNX_VART_FIRMWARE before
# /etc/vart.conf, so this points VART at the xclbin of this design.

import os

import numpy as np
from pynq import Overlay
from pynq.devicetree import DeviceTreeSegment
import errno
import fcntl
import glob
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from PIL import Image as PILImage

import cv2
import numpy as np
from IPython.display import Image, display

OVERLAY_DIR = "/root/jupyter_notebooks/YOLO"

dvt_dir = Path("/sys/kernel/config/device-tree/overlays")
protected = {"pynq"}

for overlay in dvt_dir.iterdir():
    if overlay.is_dir() and overlay.name not in protected:
        try:
            overlay.rmdir()
            print(f"Removed: {overlay.name}")
        except OSError as error:
            print(f"Could not remove {overlay.name}: {error}")

os.environ["XLNX_VART_FIRMWARE"] = f"{OVERLAY_DIR}/kv260-camonly.xclbin"

overlay = Overlay(f"{OVERLAY_DIR}/kv260-camonly.bit",
                     dtbo=f"{OVERLAY_DIR}/kv260-camonly.dtbo", ignore_version=True)

WIDTH, HEIGHT = 1920, 1080
CHECK_FRAMES = 1
OUT_FILE = os.path.join(os.getcwd(), "img.jpg")

_FMT = struct.Struct("<2I5I4H10H8I")    # struct v4l2_subdev_format, 88 bytes
_ENUM = struct.Struct("<5I7I")          # struct v4l2_subdev_mbus_code_enum, 48 bytes

def _iowr(nr, size):
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr

_ENUM_MBUS_CODE = _iowr(2, _ENUM.size)
_G_FMT = _iowr(4, _FMT.size)
_S_FMT = _iowr(5, _FMT.size)


def _find(cls, match):
    """Return /dev/<node> of the first V4L2 device in /sys/class/video4linux whose name matches."""
    for path in sorted(glob.glob(f"/sys/class/video4linux/{cls}*")):
        with open(os.path.join(path, "name")) as f:
            name = f.read().strip()
        if match(name):
            return "/dev/" + os.path.basename(path), name
    raise RuntimeError(f"No {cls} device for the camera. Load the design first: "
                       "sudo xmutil loadapp kv260-camdpu")


def _pad_fmt(fd, pad, width=None, height=None, code=None):
    """Get the active format of a pad as (width, height, code). With a code, set it first."""
    buf = bytearray(_FMT.pack(1, pad, *([0] * 27)))      # which = ACTIVE
    fcntl.ioctl(fd, _G_FMT, buf)
    if code is not None:
        v = list(_FMT.unpack(buf))
        v[2], v[3], v[4], v[5] = width, height, code, 1   # field = NONE
        buf = bytearray(_FMT.pack(*v))
        fcntl.ioctl(fd, _S_FMT, buf)
    return tuple(_FMT.unpack(buf)[2:5])


def _pad_codes(fd, pad):
    codes = []
    for index in range(64):
        buf = bytearray(_ENUM.pack(pad, index, 0, 1, 0, *([0] * 7)))
        try:
            fcntl.ioctl(fd, _ENUM_MBUS_CODE, buf)
        except OSError as e:
            if e.errno == errno.EINVAL:
                break
            raise
        codes.append(_ENUM.unpack(buf)[2])
    return codes


def set_camera_format(width, height):
    """Give the AP1302 output pad and both CSI-2 receiver pads the receiver format."""
    ap_dev, ap_name = _find("v4l-subdev", lambda n: n.startswith("ap1302"))
    csi_dev, csi_name = _find("v4l-subdev", lambda n: n.endswith(".csiss"))
    ap, csi = os.open(ap_dev, os.O_RDWR), os.open(csi_dev, os.O_RDWR)
    try:
        code = _pad_fmt(csi, 0)[2]
        if code not in _pad_codes(ap, 2):
            raise RuntimeError(f"The AP1302 driver cannot send code 0x{code:04x}.")
        got = {f"{ap_name}:2": _pad_fmt(ap, 2, width, height, code),
               f"{csi_name}:0": _pad_fmt(csi, 0, width, height, code),
               f"{csi_name}:1": _pad_fmt(csi, 1, width, height, code)}
    finally:
        os.close(ap)
        os.close(csi)
    for pad, (w, h, c) in got.items():
        print(f"  {pad:<28} code=0x{c:04x} {w}x{h}")
    if any(g != (width, height, code) for g in got.values()):
        raise RuntimeError("The pad formats do not match.")


# ---------------------------------------------------------------- capture
def _has_gstreamer():
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GStreamer:"):
            return "YES" in line
    return False


def capture_opencv(video, width, height, n_frames):
    """Read n_frames with cv2.VideoCapture through GStreamer. Return the last frame (BGR)."""
    pipeline = (f"v4l2src device={video} io-mode=mmap ! "
                f"video/x-raw,format=NV12,width={width},height={height},framerate=30/1 ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink drop=false max-buffers=2")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the camera pipeline.")
    frame = None
    try:
        for n in range(n_frames):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {n + 1} of {n_frames} could not be read.")
            frame = img
    finally:
        cap.release()
    return frame


def capture_gst_launch(video, width, height, n_frames):
    """Capture with gst-launch-1.0, keep the last raw NV12 frame, convert it with OpenCV."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "frame.nv12")
        cmd = ["gst-launch-1.0", "-q", "v4l2src", f"device={video}", f"num-buffers={n_frames}", "!",
               f"video/x-raw,format=NV12,width={width},height={height},framerate=30/1", "!",
               "multifilesink", f"location={raw}"]   # overwrites the file, so the last frame stays
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(raw):
            raise RuntimeError(f"gst-launch-1.0 failed:\n{r.stdout}{r.stderr}")
        data = np.fromfile(raw, dtype=np.uint8)
    size = width * height * 3 // 2
    if data.size < size:
        raise RuntimeError(f"The frame has {data.size} bytes. NV12 {width}x{height} needs {size}.")
    return cv2.cvtColor(data[:size].reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)


# ---------------------------------------------------------------- run

def import_image_bytes(path : str):
    with PILImage.open(path) as photo:
        gray = photo.convert("L")
        width, height = gray.size
        pixels = gray.tobytes()

    print(width, height, len(pixels))

    return pixels

def capture_image():
    video, _ = _find("video", lambda n: "vcap" in n)
    print(f"video={video}")
    set_camera_format(WIDTH, HEIGHT)

    use_opencv = _has_gstreamer()
    print(f"\nCapturing {CHECK_FRAMES} frames with "
        f"{'OpenCV + GStreamer' if use_opencv else 'gst-launch-1.0 (OpenCV has no GStreamer)'} ...")
    t0 = time.time()
    capture = capture_opencv if use_opencv else capture_gst_launch

    print("Get ready to take the picture!")
    time.sleep(1)

    print("3")
    time.sleep(1)

    print("2")
    time.sleep(1)

    print("1")
    time.sleep(1)

    frame = capture(video, WIDTH, HEIGHT, CHECK_FRAMES)
    print(f"OK. {CHECK_FRAMES} frames, frame size {frame.shape[1]}x{frame.shape[0]}.")

    # The first frames are dark until the auto exposure settles, so keep the last one.
    if not cv2.imwrite(OUT_FILE, frame):
        raise RuntimeError(f"Could not write {OUT_FILE}")
    print(f"saved {OUT_FILE}")
    display(Image(filename=OUT_FILE, width=640))

    img_bytes = import_image_bytes(OUT_FILE)

    with open("vitis/microblaze_software/img_pixels.inc", "w") as file:
        for i in range(0, len(img_bytes), 16):
            chunk = img_bytes[i:i + 16]
            file.write(", ".join(f"0x{value:02X}" for value in chunk) + ",\n")

    file.close()

    return OUT_FILE
