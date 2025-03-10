import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageSequence, ImageDraw, ImageFont
import tensorflow as tf
import os
import numpy as np
import skimage.io
from collections import Counter
import threading
import cv2  # Import OpenCV
import tifffile as tiff

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import filters

import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for Excel export
import scipy.io as sio  # Import scipy.io for saving .mat files
from skimage.measure import find_contours
import math
import time


# Import Mask RCNN
import neuroseg2.model as modellib
from neuroseg2 import visualize
from neuroseg2.config import Config

# Disable TensorFlow v2 behavior
tf.compat.v1.disable_v2_behavior()

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuring GPU settings
config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ShapeConfig(Config):
    NAME = "shape"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 192
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50


class InferenceConfig(ShapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "shape"
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0


inference_config = InferenceConfig()
inference_config.display()

# Global variables to hold model and image data
model = None
image_np = None
projected_image_np = None
max_projection_np = None
mean_projection_np = None
segmented_image_pil = None  # Global variable to hold the segmented image
masks = None  # Global variable to hold the segmentation masks
class_names = ['BG', 'object']  # Adjust this based on your classes
show_neuron_numbers = True  # Global variable to track if neuron numbers should be displayed
frame_rate = 1  # Default frame rate
region_signals = {}  # Store extracted signals
fused_image_np = None  # Global variable to hold the fused image

# Create a global TensorFlow session
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def load_model(model_path):
    global model
    with sess.as_default():
        with sess.graph.as_default():
            model = modellib.NeuroSeg(mode="inference", config=inference_config, model_dir=MODEL_DIR)
            model.load_weights(model_path, by_name=True)
            model_name = os.path.basename(model_path)
            model_name_label.configure(text="Model loaded: " + model_name)
            model_path_label.configure(text="Model path: " + model_path)


def load_model_file():
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])
    if model_path:
        progress_bar.start()
        threading.Thread(target=lambda: run_with_progress(load_model, model_path)).start()


def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.avi")])
    if file_path:
        progress_bar.start()
        threading.Thread(target=lambda: run_with_progress(process_image, file_path)).start()


def process_image(file_path):
    global image_np, projected_image_np, max_projection_np, mean_projection_np, mean_projection_enhanced_np, correlation_map_np
    clear_segmented_image()  # 每次加载新图像前清除之前的分割结果

    if file_path.lower().endswith(".tif") or file_path.lower().endswith(".tiff"):
        tif_data = read_tif(file_path)
        max_projection_np = max_projection(tif_data)
        mean_projection_np = mean_projection(tif_data)
        mean_projection_enhanced_np = mean_projection_enhanced(tif_data)
        ly, lx = tif_data.shape[1:3]
        correlation_map_np = correlation_map(tif_data, ly, lx, (0, ly), (0, lx))
        image_np = max_projection_np  # Default to max projection
        projected_image_np = max_projection_np
        image_pil = Image.fromarray(image_np)
        photo = ImageTk.PhotoImage(image_pil)
    elif file_path.lower().endswith(".avi"):
        avi_data = read_avi(file_path)
        max_projection_np = max_projection(avi_data)
        mean_projection_np = mean_projection(avi_data)
        mean_projection_enhanced_np = mean_projection_enhanced(avi_data)
        ly, lx = avi_data.shape[1:3]
        correlation_map_np = correlation_map(avi_data, ly, lx, (0, ly), (0, lx))
        image_np = max_projection_np  # Default to max projection
        projected_image_np = max_projection_np
        image_pil = Image.fromarray(image_np)
        photo = ImageTk.PhotoImage(image_pil)
    else:
        image = Image.open(file_path).convert('RGB')  # Ensure image is in RGB format
        image_np = np.array(image)
        projected_image_np = image_np
        photo = ImageTk.PhotoImage(image)

    image_label.configure(image=photo)
    image_label.image = photo
    image_path_label.configure(text="Image path: " + file_path)

def read_avi(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    return np.stack(frames, axis=0)


def display_image(image_array):
    image_pil = Image.fromarray(image_array)
    photo = ImageTk.PhotoImage(image_pil)
    image_label.configure(image=photo)
    image_label.image = photo


def convert_tif_to_image(projection_type):
    global projected_image_np, max_projection_np, mean_projection_np, mean_projection_enhanced_np, correlation_map_np, fused_image_np
    if image_np is None:
        print("No image loaded!")
        return
    progress_bar.start()
    threading.Thread(target=lambda: run_with_progress(process_tif_conversion, projection_type)).start()


def process_tif_conversion(projection_type):
    global projected_image_np
    if projection_type == 'max':
        projected_image_np = max_projection_np
    elif projection_type == 'mean':
        projected_image_np = mean_projection_np
    elif projection_type == 'mean_enhanced':
        projected_image_np = mean_projection_enhanced_np
    elif projection_type == 'correlation':
        projected_image_np = correlation_map_np
    elif projection_type == 'fuse_image':
        projected_image_np = fused_image_np
    if projected_image_np is not None:
        display_image(projected_image_np)


def read_tif(file_path):
    img = Image.open(file_path)
    frames = [np.array(frame.copy().convert("L")) for frame in ImageSequence.Iterator(img)]
    return np.stack(frames, axis=0)


# def max_projection(tif_data):
#     max_proj = np.max(tif_data, axis=0)
#     max_proj = ((max_proj - max_proj.min()) / (max_proj.max() - max_proj.min()) * 255).astype(np.uint8)
#     return max_proj

def max_projection(tif_data):
    max_proj = np.max(tif_data, axis=0)
    # 将无效值替换为最小值
    max_proj = np.nan_to_num(max_proj, nan=np.min(max_proj))
    max_proj = ((max_proj - max_proj.min()) / (max_proj.max() - max_proj.min()) * 255).astype(np.uint8)
    return max_proj


# def mean_projection(tif_data):
#     mean_proj = np.mean(tif_data, axis=0)
#     mean_proj = ((mean_proj - mean_proj.min()) / (mean_proj.max() - mean_proj.min()) * 255).astype(np.uint8)
#     return mean_proj

def mean_projection(tif_data):
    mean_proj = np.mean(tif_data, axis=0)
    # 将无效值替换为最小值
    mean_proj = np.nan_to_num(mean_proj, nan=np.min(mean_proj))
    mean_proj = ((mean_proj - mean_proj.min()) / (mean_proj.max() - mean_proj.min()) * 255).astype(np.uint8)
    return mean_proj

def mean_projection_enhanced(tif_data):
    # 计算沿第一个轴（通常是时间或帧轴）的平均值
    mimg = np.mean(tif_data, axis=0)
    # 计算平均投影图像的第1百分位数
    mimg1 = np.percentile(mimg, 1)
    # 计算平均投影图像的第99百分位数
    mimg99 = np.percentile(mimg, 99)
    # 使用第1和第99百分位数对平均投影图像进行归一化
    mimg = np.nan_to_num(mimg, nan=np.min(mimg))  # 将无效值替换为最小值 #$
    mimg = (mimg - mimg1) / (mimg99 - mimg1)
    # 将归一化后的图像限制在0到1之间
    mimg = np.maximum(0, np.minimum(1, mimg))
    # 乘以255将图像扩展到0到255范围
    mimg *= 255
    # 将图像转换为8位无符号整数
    mimg = mimg.astype(np.uint8)
    # 将单通道图像复制成三通道图像
    mimg_rgb = np.tile(mimg[:, :, np.newaxis], (1, 1, 3))
    # 返回处理后的三通道图像
    return mimg_rgb


def hp_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    mov = mov.astype(np.float32).copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov

def temporal_high_pass_filter(mov: np.ndarray, width: int) -> np.ndarray:
    mov = mov.astype(np.float32).copy()
    return hp_rolling_mean_filter(mov, width)

def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), "float32")
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix + batch_size, :, :], axis=0)**2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov

def circleMask(d0):

    dx = np.tile(np.arange(-d0[1], d0[1] + 1) / d0[1], (2 * d0[0] + 1, 1))
    dy = np.tile(np.arange(-d0[0], d0[0] + 1) / d0[0], (2 * d0[1] + 1, 1))
    dy = dy.transpose()

    rs = (dy**2 + dx**2)**0.5
    dx = dx[rs <= 1.]
    dy = dy[rs <= 1.]
    return rs, dx, dy

def getSVDdata(mov: np.ndarray, ops):
    mov = temporal_high_pass_filter(mov, width=int(ops["high_pass"]))
    ops["max_proj"] = mov.max(axis=0)
    nbins, Lyc, Lxc = np.shape(mov)
    sig = ops["diameter"] / 10.0
    for j in range(nbins):
        mov[j, :, :] = gaussian_filter(mov[j, :, :], sig)
    sdmov = standard_deviation_over_time(mov, batch_size=ops["batch_size"])
    mov /= sdmov
    mov = np.reshape(mov, (-1, Lyc * Lxc))
    cov = mov @ mov.transpose() / mov.shape[1]
    u, s, v = np.linalg.svd(cov)
    nsvd_for_roi = min(ops["nbinned"], int(cov.shape[0] / 2))
    u = u[:, :nsvd_for_roi]
    U = u.transpose() @ mov
    U = np.reshape(U, (-1, Lyc, Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()
    return ops, U, sdmov, u

# def getVmap(Ucell, sig):
#     us = gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode="wrap")
#     log_variances = (us**2).mean(axis=-1) / gaussian_filter(
#         (Ucell**2).mean(axis=-1), sig, mode="wrap")
#     return log_variances.astype("float64"), us

def getVmap(Ucell, sig):
    us = gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode="wrap")
    log_variances = (us**2).mean(axis=-1) / gaussian_filter(
        np.nan_to_num((Ucell**2).mean(axis=-1)), sig, mode="wrap")  # 将无效值替换为0
    return log_variances.astype("float64"), us

def morphOpen(V, footprint):
    vrem = filters.minimum_filter(V, footprint=footprint)
    vrem = -filters.minimum_filter(-vrem, footprint=footprint)
    return vrem

def sourcery(mov: np.ndarray, ops):
    if isinstance(ops["diameter"], int):
        ops["diameter"] = [ops["diameter"], ops["diameter"]]
    ops["diameter"] = np.array(ops["diameter"])
    ops, U, sdmov, u = getSVDdata(mov=mov, ops=ops)
    sig = np.ceil(ops["diameter"] / 4)
    V, us = getVmap(U, sig)

    d0 = ops["diameter"]
    rs, dy, dx = circleMask(d0)

    vrem = morphOpen(V, rs <= 1.)
    V = V - vrem
    V = V.astype("float64")
    maxV = filters.maximum_filter(V, footprint=np.ones((3, 3)), mode="reflect")
    imax = V > (maxV - 1e-10)
    peaks = V[imax]
    thres = ops["threshold_scaling"] * np.median(peaks[peaks > 1e-4])
    ops["Vcorr"] = V
    return ops

def correlation_map(tif_data, ly, lx, yrange, xrange):
    ops = {
        "diameter": [10, 10],
        "threshold_scaling": 0.5,
        "high_pass": 50,
        "batch_size": 500,
        "nbinned": 5000,
    }
    ops = sourcery(tif_data, ops)
    vcorr = ops["Vcorr"]
    mimg1 = np.percentile(vcorr, 1)
    mimg99 = np.percentile(vcorr, 99)
    vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
    mimg = mimg1 * np.ones((ly, lx), np.float32)
    mimg[yrange[0]:yrange[1], xrange[0]:xrange[1]] = vcorr
    mimg = np.maximum(0, np.minimum(1, mimg))
    mimg *= 255
    mimg = mimg.astype(np.uint8)
    mimg_rgb = np.tile(mimg[:, :, np.newaxis], (1, 1, 3))
    return mimg_rgb


def fuse_images():
    global mean_projection_enhanced_np, correlation_map_np, fused_image_np, projected_image_np
    try:
        mean_projection_coeff = float(mean_projection_entry.get()) / 100.0
        correlation_map_coeff = float(correlation_map_entry.get()) / 100.0

        if mean_projection_np is None or correlation_map_np is None:
            print("Mean projection or correlation map not loaded!")
            return

        fused_image_np = cv2.addWeighted(mean_projection_enhanced_np, mean_projection_coeff,
                                         correlation_map_np, correlation_map_coeff, 0)
        projected_image_np = fused_image_np  # 更新 projected_image_np
        display_image(fused_image_np)
    except Exception as e:
        print("An error occurred during image fusion:", str(e))



def segment_image():
    global model, projected_image_np
    if model is None or projected_image_np is None:
        print("Model or image not loaded!")
        return
    progress_bar.start()
    threading.Thread(target=run_with_progress, args=(perform_segmentation,)).start()


def clear_segmented_image():
    global segmented_image_pil, segmented_image_label, masks, projected_image_np, fused_image_np
    segmented_image_pil = None
    segmented_image_label.config(image='')
    neuron_count_label.config(text="Neuron Counts:\n")
    masks = None
    projected_image_np = None
    fused_image_np = None


def perform_segmentation():
    global model, projected_image_np, segmented_image_pil, masks, fused_image_np
    image_to_segment = fused_image_np if fused_image_np is not None else projected_image_np

    # If the image is grayscale, convert it to RGB
    if len(image_to_segment.shape) == 2:
        image_to_segment = np.stack((image_to_segment,) * 3, axis=-1)

    image_np_expanded = np.expand_dims(image_to_segment, axis=0).astype(np.float32)
    try:
        with sess.as_default():
            with sess.graph.as_default():
                results = model.detect(image_np_expanded, verbose=1)
                r = results[0]
                masks = r['masks']  # Store masks for signal extraction

                # Apply masks to the image and draw region numbers
                masked_image = image_to_segment.copy().astype(np.uint8)
                masked_image_pil = Image.fromarray(masked_image)
                draw = ImageDraw.Draw(masked_image_pil)
                font = ImageFont.load_default()
                for i in range(masks.shape[-1]):
                    mask = masks[:, :, i]
                    mask_resized = cv2.resize(mask.astype(np.uint8), (image_to_segment.shape[1], image_to_segment.shape[0]), interpolation=cv2.INTER_NEAREST)
                    masked_image = visualize.apply_mask(np.array(masked_image_pil), mask_resized, (1.0, 0.0, 0.0), alpha=0.3)  # 调整透明度
                    masked_image_pil = Image.fromarray(masked_image)
                    if show_neuron_numbers:
                        # Find the center of the mask
                        y, x = np.where(mask_resized)
                        center_y, center_x = int(np.mean(y)), int(np.mean(x))
                        draw.text((center_x, center_y), str(i + 1), fill="white", font=font)

                # Convert the masked image to PIL format
                segmented_image_pil = masked_image_pil
                photo = ImageTk.PhotoImage(segmented_image_pil)
                segmented_image_label.configure(image=photo)
                segmented_image_label.image = photo

                # Print class counts and update neuron count label
                class_ids = r['class_ids']
                class_counts = Counter(class_ids)
                count_text = "\n".join(
                    [f"Neuron: {i + 1}, Count: {count}" for i, count in enumerate(class_counts.values())])
                neuron_count_label.configure(text="Neuron Counts:\n" + count_text)

    except Exception as e:
        print("An error occurred during segmentation:", str(e))


def extract_signals():
    file_path = filedialog.askopenfilename(filetypes=[("TIF Files", "*.tif;*.tiff")])
    if file_path and masks is not None:
        progress_bar.start()
        threading.Thread(target=lambda: run_with_progress(process_signal_extraction, file_path)).start()


def process_signal_extraction(file_path):
    global masks, frame_rate, region_signals
    try:
        image = tiff.imread(file_path)
        num_frames, height, width = image.shape
        signals = np.zeros((height, width, num_frames))

        for y in range(height):
            for x in range(width):
                pixel_signal = image[:, y, x]
                # Apply Gaussian filter for denoising
                sigma = 1
                pixel_signal = gaussian_filter(pixel_signal, sigma=sigma)
                signals[y, x, :] = pixel_signal

        # Extract signals from masks
        region_signals = {}
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            region_signal = np.mean([signals[y, x, :] for y in range(height) for x in range(width) if mask[y, x]],
                                    axis=0)
            region_signals[f'Neuron {i + 1}'] = region_signal

        # Visualize signals
        neurons_to_display = list(region_signals.items())
        signal_plots = []
        total_neurons = len(region_signals)
        pages = total_neurons // 5 + (1 if total_neurons % 5 != 0 else 0)
        time_axis = np.arange(num_frames) / frame_rate  # Convert frame number to time in seconds
        for page in range(pages):
            start_index = page * 5
            end_index = start_index + 5
            neurons_to_display = list(region_signals.items())[start_index:end_index]

            fig, axs = plt.subplots(len(neurons_to_display), 1, figsize=(20, 8 * len(neurons_to_display)))
            if len(neurons_to_display) == 1:
                axs = [axs]
            for i, (neuron, signal) in enumerate(neurons_to_display):
                axs[i].plot(time_axis, signal, linewidth=1.5)  # 加粗神经元信号
                axs[i].set_xlabel('Time (s)', fontsize=12)
                axs[i].set_ylabel('Signal', fontsize=12)
                axs[i].set_title(f"{neuron}", fontsize=14)
            plt.tight_layout()
            signal_plots.append(fig)

        # Display all signal plots
        for fig in signal_plots:
            plt.show(fig)
            plt.close(fig)  # Close the figure after showing it to free up memory

    except Exception as e:
        print("An error occurred during signal extraction:", str(e))


def save_signals_to_excel():
    global region_signals
    if not region_signals:
        print("No signals to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    if save_path:
        try:
            df = pd.DataFrame(region_signals)
            df.to_excel(save_path, index=False, engine='openpyxl')
            print(f"Signals saved to {save_path}")
        except Exception as e:
            print(f"An error occurred while saving signals: {str(e)}")


def save_image():
    global projected_image_np
    if projected_image_np is None:
        print("No image to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        Image.fromarray(projected_image_np).save(save_path)


def save_segmented_image():
    global segmented_image_pil
    if segmented_image_pil is None:
        print("No segmented image to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        segmented_image_pil.save(save_path)


def toggle_neuron_numbers():
    global show_neuron_numbers
    show_neuron_numbers = not show_neuron_numbers
    if projected_image_np is not None and segmented_image_pil is not None:
        update_segmented_image()  # Update segmentation image to reflect the change


def update_segmented_image():
    global segmented_image_pil, masks, projected_image_np, show_neuron_numbers
    if masks is not None and projected_image_np is not None:
        # 使用原始图像
        masked_image = projected_image_np.copy().astype(np.uint8)

        # 如果图像是灰度图像，将其转换为三通道
        if len(masked_image.shape) == 2:
            masked_image = np.stack((masked_image,) * 3, axis=-1)

        for i in range(masks.shape[-1]):
            color = visualize.random_colors(1)[0]
            mask = masks[:, :, i]
            mask_resized = cv2.resize(mask.astype(np.uint8), (masked_image.shape[1], masked_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            masked_image = visualize.apply_mask(masked_image, mask_resized, color, alpha=0.3)  # 调整透明度
            if show_neuron_numbers:
                centroid_y, centroid_x = np.mean(np.where(mask_resized), axis=1).astype(int)
                cv2.putText(masked_image, str(i + 1), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        segmented_image_pil = Image.fromarray(masked_image.astype(np.uint8))
        photo = ImageTk.PhotoImage(segmented_image_pil)
        segmented_image_label.configure(image=photo)
        segmented_image_label.image = photo



def save_segmented_results_to_mat():
    global masks, projected_image_np
    if masks is None or projected_image_np is None:
        print("No segmentation results to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".mat",
                                             filetypes=[("MAT files", "*.mat"), ("All files", "*.*")])
    if save_path:
        try:
            xypos = []
            xy_all = []
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                contours = find_contours(mask, 0.5)
                if contours:
                    contour = contours[0]
                    x = contour[:, 1]
                    y = contour[:, 0]
                    xypos.append(np.array([x, y]).tolist())  # 存储每个分割神经元的闭合边缘坐标
                    xy_all.append([float(np.mean(x)), float(np.mean(y))])  # 存储每个分割神经元的质心坐标

            ParametersOutput = {
                'xypos1': xypos,
                'xy_all1': xy_all,
                'Pixels1': [[list(pair) for pair in zip(np.where(mask)[0], np.where(mask)[1])] for mask in
                           masks.transpose(2, 0, 1)]
            }

            ImageDataMean = projected_image_np[:, :, 0].astype(np.uint8)  # Ensure it's 512x512 and uint8

            hline = np.zeros((1, len(xypos)), dtype=object)  # Placeholder for hline object
            htext = np.zeros((1, len(xypos)), dtype=object)  # Placeholder for htext object

            sio.savemat(save_path, {
                'ImageDataMean1': ImageDataMean,
                'ParametersOutput1': ParametersOutput,
                'hline1': hline,
                'htext1': htext
            })
            print(f"Segmentation results saved to {save_path}")
        except Exception as e:
            print(f"An error occurred while saving segmentation results: {str(e)}")


def run_with_progress(func, *args):
    func(*args)
    progress_bar.stop()


# Create main window
window = tk.Tk()
window.title("NeuroSeg2")
window.geometry("1280x720")

# Create frames for layout
left_frame = tk.Frame(window)
left_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

right_frame = tk.Frame(window)
right_frame.pack(side="right", padx=10, pady=10, fill="y")

left_top_frame = tk.Frame(left_frame)
left_top_frame.pack(side="top", padx=10, pady=10)

neuron_count_label = tk.Label(left_top_frame, text="Neuron Counts:\n")
neuron_count_label.pack(side="top", padx=10, pady=10)

image_label = tk.Label(left_frame)
image_label.pack(side="left", padx=10, pady=10, expand=True)

segmented_image_label = tk.Label(left_frame)
segmented_image_label.pack(side="left", padx=10, pady=10, expand=True)

# Create buttons for loading image and model
select_image_button = tk.Button(right_frame, text="Load Image", command=load_image)
select_image_button.pack(padx=10, pady=10)

image_path_label = tk.Label(right_frame, text="")
image_path_label.pack(padx=10, pady=5)

select_model_button = tk.Button(right_frame, text="Load Model", command=load_model_file)
select_model_button.pack(padx=10, pady=10)

model_path_label = tk.Label(right_frame, text="")
model_path_label.pack(padx=10, pady=5)

button_frame = tk.Frame(right_frame)
button_frame.pack(padx=10, pady=10)

convert_max_tif_button = tk.Button(button_frame, text="Max Projection", command=lambda: convert_tif_to_image('max'))
convert_max_tif_button.grid(row=0, column=0, padx=10, pady=10)

convert_mean_tif_button = tk.Button(button_frame, text="Mean Projection", command=lambda: convert_tif_to_image('mean'))
convert_mean_tif_button.grid(row=0, column=1, padx=10, pady=10)

convert_mean_enhanced_tif_button = tk.Button(button_frame, text="Mean Projection Enhanced",
                                             command=lambda: convert_tif_to_image('mean_enhanced'))
convert_mean_enhanced_tif_button.grid(row=1, column=0, padx=10, pady=10)

convert_correlation_map_button = tk.Button(button_frame, text="Correlation Map",
                                           command=lambda: convert_tif_to_image('correlation'))
convert_correlation_map_button.grid(row=1, column=1, padx=10, pady=10)

# Fusion coefficients input
fusion_coefficients_label = tk.Label(right_frame, text="Fusion Coefficients:")
fusion_coefficients_label.pack(padx=10, pady=5)

fusion_frame = tk.Frame(right_frame)
fusion_frame.pack(padx=10, pady=5)

main_image_label = tk.Label(fusion_frame, text="The Main Image:")
main_image_label.grid(row=0, column=0, padx=10, pady=5)
mean_projection_entry = tk.Entry(fusion_frame)
mean_projection_entry.grid(row=0, column=1, padx=10, pady=5)
mean_projection_entry.insert(0, "50")  # Default value for mean projection coefficient

correlation_map_label = tk.Label(fusion_frame, text="Correlation Map:")
correlation_map_label.grid(row=1, column=0, padx=10, pady=5)
correlation_map_entry = tk.Entry(fusion_frame)
correlation_map_entry.grid(row=1, column=1, padx=10, pady=5)
correlation_map_entry.insert(0, "50")  # Default value for correlation map coefficient

fuse_button = tk.Button(right_frame, text="Fuse Images", command=fuse_images)
fuse_button.pack(padx=10, pady=10)

# Create button for segmenting image
segment_button = tk.Button(right_frame, text="Segment Image", command=segment_image)
segment_button.pack(padx=10, pady=10)

# Create button for clearing segmented image
clear_button = tk.Button(right_frame, text="Clear Image", command=clear_segmented_image)
clear_button.pack(padx=10, pady=10)

# Create button for extracting signals
extract_signals_button = tk.Button(right_frame, text="Extract Signals", command=extract_signals)
extract_signals_button.pack(padx=10, pady=10)

# Add input for frame rate
frame_rate_label = tk.Label(right_frame, text="Frame rate (Hz):")
frame_rate_label.pack(padx=10, pady=5)
frame_rate_entry = tk.Entry(right_frame)
frame_rate_entry.pack(padx=10, pady=5)
frame_rate_entry.insert(0, "1")  # Default value

def set_frame_rate():
    global frame_rate
    frame_rate = float(frame_rate_entry.get())

# Create button to set frame rate
set_frame_rate_button = tk.Button(right_frame, text="Set Frame Rate", command=set_frame_rate)
set_frame_rate_button.pack(padx=10, pady=10)

# Create frame for saving buttons
save_button_frame = tk.Frame(right_frame)
save_button_frame.pack(padx=10, pady=10)

# Create button for saving signals to Excel
save_signals_button = tk.Button(save_button_frame, text="Save Signals", command=save_signals_to_excel)
save_signals_button.grid(row=0, column=0, padx=10, pady=10)

# Create button for saving images
save_image_button = tk.Button(save_button_frame, text="Save Image", command=save_image)
save_image_button.grid(row=0, column=1, padx=10, pady=10)

# Create button for saving segmented images
save_segmented_button = tk.Button(save_button_frame, text="Save Segmented Image", command=save_segmented_image)
save_segmented_button.grid(row=1, column=0, padx=10, pady=10)

# Create button for saving segmented results to .mat file
save_mat_button = tk.Button(save_button_frame, text="Save Results to MAT", command=save_segmented_results_to_mat)
save_mat_button.grid(row=1, column=1, padx=10, pady=10)

# Create buttons to toggle neuron numbers
toggle_numbers_button = tk.Button(right_frame, text="Toggle Neuron Numbers", command=toggle_neuron_numbers)
toggle_numbers_button.pack(padx=10, pady=10)

# Create label for displaying model name
model_name_label = tk.Label(right_frame, text="Model not loaded")
model_name_label.pack(padx=10, pady=10)

# Create progress bar
progress_bar = ttk.Progressbar(right_frame, mode='indeterminate')
progress_bar.pack(padx=10, pady=20, fill='x')

# Run the main loop
window.mainloop()

