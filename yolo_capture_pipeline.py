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
import time
import cv2
import random
import colorsys
import errno
import fcntl
import glob
import struct
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from pynq_dpu import DpuOverlay
from pynq.lib.video import DisplayPort, VideoMode, PIXEL_RGB

OVERLAY_DIR = "/root/jupyter_notebooks/YOLO"
XMODEL = "/root/jupyter_notebooks/pynq-dpu/tf_yolov3_voc.xmodel"

os.environ["XLNX_VART_FIRMWARE"] = f"{OVERLAY_DIR}/kv260-camdpu.xclbin"

overlay = DpuOverlay(f"{OVERLAY_DIR}/kv260-camdpu.bit",
                     dtbo=f"{OVERLAY_DIR}/kv260-camdpu.dtbo")
overlay.load_model(XMODEL)
dpu = overlay.runner

inputs = dpu.get_input_tensors()
outputs = dpu.get_output_tensors()
print("inputs: ", [tuple(t.dims) for t in inputs])
print("outputs:", [tuple(t.dims) for t in outputs])

# One test inference with zero input. Use the dtype that this pynq_dpu /
# VART version expects: the PYNQ YOLOv3 notebook uses float32.
in_buf = [np.zeros(tuple(t.dims), dtype=np.float32, order="C") for t in inputs]
out_buf = [np.empty(tuple(t.dims), dtype=np.float32, order="C") for t in outputs]
job = dpu.execute_async(in_buf, out_buf)
dpu.wait(job)
print("DPU inference OK")

class YoloV3FpgaInference:
    def __init__(self):
        pass
    
    def initialize(self, overlay, runner, classes_path="img/voc_classes.txt"):
        # The overlay and the runner are made in the load cell (top level, as in
        # TestDpu.ipynb). The class only keeps references to them.
        self.overlay = overlay
        self.dpu = runner

        anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
        anchor_float = [float(x) for x in anchor_list]
        self.anchors = np.array(anchor_float).reshape(-1, 2)
        self.class_names = self.get_class(classes_path)
        num_classes = len(self.class_names)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x:
                          (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                          colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        self.input_tensors = self.dpu.get_input_tensors()
        self.output_tensors = self.dpu.get_output_tensors()
        inputTensors = self.input_tensors
        outputTensors = self.output_tensors
        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = (tuple(outputTensors[0].dims)) # (1, 13, 13, 75)
        self.shapeOut1 = (tuple(outputTensors[1].dims)) # (1, 26, 26, 75)
        self.shapeOut2 = (tuple(outputTensors[2].dims)) # (1, 52, 52, 75)
        self.outputSize0 = int(outputTensors[0].get_data_size() / self.shapeIn[0]) # 12675
        self.outputSize1 = int(outputTensors[1].get_data_size() / self.shapeIn[0]) # 50700
        self.outputSize2 = int(outputTensors[2].get_data_size() / self.shapeIn[0]) # 202800
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [np.empty(self.shapeOut0, dtype=np.float32, order="C"),
                       np.empty(self.shapeOut1, dtype=np.float32, order="C"),
                       np.empty(self.shapeOut2, dtype=np.float32, order="C")]
        self.image = self.input_data[0]
        print(f"DPU ready: input {self.shapeIn}, "
              f"outputs {self.shapeOut0} {self.shapeOut1} {self.shapeOut2}")

    def get_class(self, classes_path):
        with open(classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]
        return self.class_names
    
    '''resize image with unchanged aspect ratio using padding'''
    def letterbox_image(self, image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w/iw, h/ih)
        #print(scale)

        nw = int(iw*scale)
        nh = int(ih*scale)
        #print(nw)
        #print(nh)

        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.ones((h,w,3), np.uint8) * 128
        h_start = (h-nh)//2
        w_start = (w-nw)//2
        new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
        return new_image


    '''image preprocessing'''
    def pre_process(self, image, model_image_size):
        image = image[...,::-1]
        image_h, image_w, _ = image.shape

        if model_image_size != (None, None):
            assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0) 	
        return image_data
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        grid_size = np.shape(feats)[1:3]
        nu = num_classes + 5
        predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
        grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis = -1)
        grid = np.array(grid, dtype=np.float32)

        box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
        box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
        box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
        box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
        return box_xy, box_wh, box_confidence, box_class_probs


    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape, dtype = np.float32)
        image_shape = np.array(image_shape, dtype = np.float32)
        new_shape = np.around(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)
        return boxes


    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores


    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.55)[0]  # threshold
            order = order[inds + 1]

        return keep
    
    def draw_boxes(self, frame, boxes, scores, classes):
        # _, ax = plt.subplots(1)
        # ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_h, image_w, _ = frame.shape

        for i, bbox in enumerate(boxes):
            [top, left, bottom, right] = bbox
            color = (255, 255, i*255//len(boxes))
            # width, height = right - left, bottom - top
            # center_x, center_y = left + width*0.5, top + height*0.5
            # score, class_index = scores[i], classes[i]
            top_left = (int(left), int(top))
            bottom_right = (int(right), int(bottom))
            cv2.rectangle(frame, top_left, bottom_right, color, 1)
            cv2.putText(frame, self.class_names[classes[i]], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # label = '{}: {:.4f}'.format(self.class_names[class_index], score)
            # color = tuple([color/255 for color in colors[class_index]])
            # ax.add_patch(Rectangle((left, top), width, height,
            #                        edgecolor=color, facecolor='none'))
            # ax.annotate(label, (center_x, center_y), color=color, weight='bold', 
            #             fontsize=12, ha='center', va='center')
        # return ax
    
    def evaluate(self, yolo_outputs, image_shape, class_names, anchors):
        score_thresh = 0.2
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = np.shape(yolo_outputs[0])[1 : 3]
        input_shape = np.array(input_shape)*32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i], anchors[anchor_mask[i]], len(self.class_names), 
                input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis = 0)
        box_scores = np.concatenate(box_scores, axis = 0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(self.class_names)):
            class_boxes_np = boxes[mask[:, c]]
            class_box_scores_np = box_scores[:, c]
            class_box_scores_np = class_box_scores_np[mask[:, c]]
            nms_index_np = self.nms_boxes(class_boxes_np, class_box_scores_np) 
            class_boxes_np = class_boxes_np[nms_index_np]
            class_box_scores_np = class_box_scores_np[nms_index_np]
            classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
            boxes_.append(class_boxes_np)
            scores_.append(class_box_scores_np)
            classes_.append(classes_np)
        boxes_ = np.concatenate(boxes_, axis = 0)
        scores_ = np.concatenate(scores_, axis = 0)
        classes_ = np.concatenate(classes_, axis = 0)

        return boxes_, scores_, classes_
        
    def run(self, frame):
        # Pre-processing
        image_size = frame.shape[:2]
        image_data = np.array(self.pre_process(frame, (416, 416)), dtype=np.float32)
        # Fetch data to DPU and trigger it
        self.image[0,...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        # Retrieve output data
        self.conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        self.conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        self.conv_out2 = np.reshape(self.output_data[2], self.shapeOut2)
        yolo_outputs = [self.conv_out0, self.conv_out1, self.conv_out2]

        # Decode output from YOLOv3
        boxes, scores, classes = self.evaluate(yolo_outputs, image_size, self.class_names, self.anchors)

        self.draw_boxes(frame, boxes, scores, classes)
        # print("Number of detected objects: {}".format(len(boxes)))
        # print(boxes)
        # print(scores)
        # print(classes)
        
        return frame

# ---------------------------------------------------------------- pad formats
# The camera path is YUV 4:2:0 (MEDIA_BUS_FMT_VYYUYY8_1X24, code 0x2100,
# Xilinx specific). media-ctl does not know the name, so set the pad formats
# with the V4L2 subdev ioctls directly (same code as TestCam.ipynb).
_FMT = struct.Struct("<2I5I4H10H8I")    # struct v4l2_subdev_format, 88 bytes
_ENUM = struct.Struct("<5I7I")          # struct v4l2_subdev_mbus_code_enum, 48 bytes


def _iowr(nr, size):
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


_ENUM_MBUS_CODE = _iowr(2, _ENUM.size)
_G_FMT = _iowr(4, _FMT.size)
_S_FMT = _iowr(5, _FMT.size)


def _find(cls, match):
    """Return (/dev/<node>, name) of the first V4L2 device whose name matches."""
    for path in sorted(glob.glob(f"/sys/class/video4linux/{cls}*")):
        with open(os.path.join(path, "name")) as f:
            name = f.read().strip()
        if match(name):
            return "/dev/" + os.path.basename(path), name
    raise RuntimeError(f"No {cls} device for the camera. Load the overlay first "
                       "(run the load cell, the first code cell).")


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
        got = [_pad_fmt(ap, 2, width, height, code),
               _pad_fmt(csi, 0, width, height, code),
               _pad_fmt(csi, 1, width, height, code)]
    finally:
        os.close(ap)
        os.close(csi)
    if any(g != (width, height, code) for g in got):
        raise RuntimeError(f"The pad formats do not match: {got}")


# ---------------------------------------------------------------- frames
# Cameras that are still open. A new MipiCamera closes them first, because the
# capture process of a forgotten camera keeps /dev/video0 busy.
# setdefault: re-running this cell keeps the list.
_OPEN_CAMERAS = globals().setdefault("_OPEN_CAMERAS", [])


class MipiCamera:
    """Continuous capture from the MIPI camera. read() returns a BGR frame.

    The OpenCV in the PYNQ environment has no GStreamer support, so one
    gst-launch-1.0 process streams raw NV12 frames into a pipe. A leaky queue
    of one frame drops old frames when the reader is slower than the camera,
    so read() always gets a recent frame.
    """

    def __init__(self, width=1920, height=1080, fps=30):
        for cam in list(_OPEN_CAMERAS):
            print(f"Closing a camera that was left open ({cam.device})")
            cam.close()
        self.width, self.height = width, height
        self.device, _ = _find("video", lambda n: "vcap" in n)
        set_camera_format(width, height)
        self.frame = np.empty(width * height * 3 // 2, dtype=np.uint8)
        self._log = tempfile.TemporaryFile()
        cmd = ["gst-launch-1.0", "-q",
               "v4l2src", f"device={self.device}", "io-mode=mmap", "!",
               f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1", "!",
               "queue", "max-size-buffers=1", "leaky=downstream", "!",
               "fdsink", "fd=1", "sync=false"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._log, bufsize=0)
        _OPEN_CAMERAS.append(self)

    def read(self):
        view, got = memoryview(self.frame), 0
        while got < len(view):
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                self._log.seek(0)
                raise RuntimeError("Camera stream stopped:\n" + self._log.read().decode(errors="replace"))
            got += n
        return cv2.cvtColor(self.frame.reshape(self.height * 3 // 2, self.width),
                            cv2.COLOR_YUV2BGR_NV12)

    def close(self):
        if self in _OPEN_CAMERAS:
            _OPEN_CAMERAS.remove(self)
        if self.proc.stdout.closed:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()
        self._log.close()


class CameraPipeline:
    """Camera -> callback -> output. Subclasses implement the output in _write()."""

    def __init__(self, frame_in_w=1920, frame_in_h=1080, frame_out_w=1920, frame_out_h=1080):
        self.frame_in_w = frame_in_w
        self.frame_in_h = frame_in_h
        self.frame_out_w = frame_out_w
        self.frame_out_h = frame_out_h
        self.camera = MipiCamera(frame_in_w, frame_in_h)
        self.frame_times = []
        self._closed = False
        print(f"Camera {self.camera.device} open: {frame_in_w}x{frame_in_h}")

    def run(self, iterations, callback):
        local_frame_times = []
        for _ in range(iterations):
            start_frame_time = time.perf_counter()
            frame = callback(self.camera.read())
            if frame.shape[1] != self.frame_out_w or frame.shape[0] != self.frame_out_h:
                frame = cv2.resize(frame, (self.frame_out_w, self.frame_out_h))
            self._write(frame)
            local_frame_times.append(time.perf_counter() - start_frame_time)
        self.frame_times.append(local_frame_times)

    def _write(self, frame):
        raise NotImplementedError

    def _close_output(self):
        pass

    def plot_frame_times(self, labels):
        sums = []
        for i, frame_times in enumerate(self.frame_times):
            y = [round(frame_time * 1000) for frame_time in frame_times]
            x = list(range(len(frame_times)))
            plt.semilogy(x, y, base=2, marker="o", linestyle="-", label=labels[i])
            sums.append(sum(frame_times))
        plt.xlabel("Frame")
        plt.ylabel("Time (ms)")
        plt.title("DPU vs CPU")
        plt.legend()
        plt.show()
        for i, frame_times in enumerate(self.frame_times):
            print(f"{labels[i]}: {len(frame_times) / sums[i]:.2f}fps")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.camera.close()
        self._close_output()


class CameraToDisplayport(CameraPipeline):
    """Shows the frames on the DisplayPort output."""

    def __init__(self, frame_in_w=1920, frame_in_h=1080, frame_out_w=1920, frame_out_h=1080):
        super().__init__(frame_in_w, frame_in_h, frame_out_w, frame_out_h)
        self.displayport = DisplayPort()
        self.displayport.configure(VideoMode(frame_out_w, frame_out_h, 24), PIXEL_RGB)

    def _write(self, frame):
        outframe = self.displayport.newframe()
        outframe[:] = frame
        self.displayport.writeframe(outframe)

    def _close_output(self):
        self.displayport.stop()

class YoloV3CpuInference:

    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow("yolov3_voc.pb")

        self.INPUT_SIZE = 416
        self.anchors = np.array([
            [10,13], [16,30], [33,23],
            [30,61], [62,45], [59,119],
            [116,90], [156,198], [373,326]
        ], dtype=np.float32)

        self.anchor_masks = [
            [6,7,8],
            [3,4,5],
            [0,1,2]
        ]

        self.classes = [
            "aeroplane","bicycle","bird","boat","bottle",
            "bus","car","cat","chair","cow",
            "diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor"
        ]

        self.num_classes = len(self.classes)


    def run(self, img):
        blob_img, scale, dx, dy = self.letterbox_image(img, self.INPUT_SIZE)

        blob = cv2.dnn.blobFromImage(
            blob_img, 1/255.0,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            swapRB=True
        )

        self.net.setInput(blob)

        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)

        boxes, confidences, class_ids = self.process_outputs(
            outputs, img.shape, scale, dx, dy
        )

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.45)

        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                label = self.classes[class_ids[i]]
                conf = confidences[i]

                cv2.rectangle(img, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

        return img


    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))


    def letterbox_image(self, img, size):
        h, w = img.shape[:2]
        scale = min(size/w, size/h)
        nw, nh = int(w*scale), int(h*scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((size, size, 3), 128, dtype=np.uint8)

        dx = (size - nw) // 2
        dy = (size - nh) // 2

        canvas[dy:dy+nh, dx:dx+nw] = resized
        return canvas, scale, dx, dy


    def decode_output(self, output, anchors_masked):
        output = output[0]

        if output.shape[0] == len(anchors_masked) * (5 + self.num_classes):
            output = np.transpose(output, (1, 2, 0))

        grid_h, grid_w, channels = output.shape
        num_anchors = len(anchors_masked)

        expected_channels = num_anchors * (5 + self.num_classes)
        assert channels == expected_channels

        output = output.reshape(
            grid_h, grid_w, num_anchors, 5 + self.num_classes
        )

        box_xy = self.sigmoid(output[..., 0:2])
        box_wh = np.exp(output[..., 2:4])
        obj_conf = self.sigmoid(output[..., 4])
        class_probs = self.sigmoid(output[..., 5:])

        grid_x = np.arange(grid_w)
        grid_y = np.arange(grid_h)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        grid = np.stack((grid_x, grid_y), axis=-1)
        grid = np.expand_dims(grid, axis=2)

        box_xy = (box_xy + grid) / [grid_w, grid_h]

        anchors_masked = np.array(anchors_masked, dtype=np.float32)
        box_wh = (box_wh * anchors_masked) / self.INPUT_SIZE

        return box_xy, box_wh, obj_conf, class_probs

    def process_outputs(self, outputs, img_shape, scale, dx, dy):
        boxes = []
        confidences = []
        class_ids = []

        for i, output in enumerate(outputs):
            anchors_masked = self.anchors[self.anchor_masks[i]]

            box_xy, box_wh, obj_conf, class_probs = self.decode_output(
                output, anchors_masked
            )

            scores = obj_conf[..., np.newaxis] * class_probs

            for y in range(box_xy.shape[0]):
                for x in range(box_xy.shape[1]):
                    for a in range(len(anchors_masked)):

                        class_id = np.argmax(scores[y, x, a])
                        score = scores[y, x, a, class_id]

                        if score < 0.5:
                            continue

                        bx, by = box_xy[y, x, a]
                        bw, bh = box_wh[y, x, a]

                        x1 = (bx - bw/2) * self.INPUT_SIZE
                        y1 = (by - bh/2) * self.INPUT_SIZE
                        x2 = (bx + bw/2) * self.INPUT_SIZE
                        y2 = (by + bh/2) * self.INPUT_SIZE

                        x1 = (x1 - dx) / scale
                        x2 = (x2 - dx) / scale
                        y1 = (y1 - dy) / scale
                        y2 = (y2 - dy) / scale

                        x1 = int(max(0, x1))
                        y1 = int(max(0, y1))
                        x2 = int(min(img_shape[1], x2))
                        y2 = int(min(img_shape[0], y2))

                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        confidences.append(float(score))
                        class_ids.append(class_id)

        return boxes, confidences, class_ids

def notebook_stream():
    yolodpu = YoloV3FpgaInference()
    yolodpu.initialize(overlay, dpu)      # overlay and dpu come from the load cell
    print("Loading CPU model (yolov3_voc.pb, 246 MB) ...", flush=True)
    t0 = time.time()
    yolocpu = YoloV3CpuInference()
    print(f"CPU model loaded in {time.time() - t0:.1f} s")

    camera_to_displayport = CameraToDisplayport()
    try:
        camera_to_displayport.run(5, yolocpu.run)
        camera_to_displayport.run(5, yolodpu.run)
    finally:
        camera_to_displayport.close()
    camera_to_displayport.plot_frame_times(["CPU", "DPU"])