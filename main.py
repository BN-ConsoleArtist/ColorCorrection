import sys

import rawpy
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_checker_detection import detect_colour_checkers_segmentation
from colour import (RGB_COLOURSPACES, CCS_COLOURCHECKERS, matrix_colour_correction, apply_matrix_colour_correction,
                    xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_Lab, cctf_encoding, delta_E)
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QPlainTextEdit, QLineEdit, QLabel, QApplication, QMainWindow, QPushButton,
                               QFileDialog, QVBoxLayout, QHBoxLayout, QBoxLayout, QWidget, QStyle)


def raw_to_rgb(image_path):
    # Convert raw .CR2 files to a float32 linear sRGB NumPy Array
    with rawpy.imread(str(image_path)) as raw:
        rgb_uint16 = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,

                # disable rawpy white balancing
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],  # Forces a 1.0 multiplier across R, G, B, and G2

                # disable rawpy CCM
                output_color=rawpy.ColorSpace.raw
            )  # uint16 numpy array [x,y,[r,g,b]]

        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to linear float32, normalize rgb values
    return rgb_float32

def calculate_wb_weights_simple(target_patches):
    # White balance patches based off 'Color Checker Classic 24 After 2014' Neutral 5 (Patch 22)
    neutral5_grey = target_patches[21]
    epsilon = 0.00001
    neutral5_reflectance_rgb = np.array([0.190087, 0.19086, 0.189828]) # patch 22 reference values
    weights_simple_rgb = neutral5_reflectance_rgb / np.maximum(neutral5_grey, epsilon)

    return weights_simple_rgb

def calculate_wb_weights_llsr(target_patch):
    def calculate_absolute_gain(target, reference):
        # Linear least-squares gain, finding a common weight for all 6 neutral patches
        return np.sum(target * reference) / np.sum(target ** 2)

    # Official post-2014 per-channel neutral reference targets (Patches 19 to 24)
    reference_r = np.array([0.879191, 0.584440, 0.357678, 0.190087, 0.085935, 0.031360])
    reference_g = np.array([0.884767, 0.592124, 0.367060, 0.190860, 0.088738, 0.031500])
    reference_b = np.array([0.834953, 0.584582, 0.365287, 0.189828, 0.089788, 0.032311])

    # Sampled linear RGB averages from the camera raw image [Patches 19 -> 24]
    target_r = np.array([target_patch[18][0], target_patch[19][0],target_patch[20][0],
                    target_patch[21][0], target_patch[22][0], target_patch[23][0]
                    ])
    target_g = np.array([target_patch[18][1], target_patch[19][1], target_patch[20][1],
                    target_patch[21][1], target_patch[22][1], target_patch[23][1]
                    ])
    target_b = np.array([target_patch[18][2], target_patch[19][2],target_patch[20][2],
                    target_patch[21][2], target_patch[22][2], target_patch[23][2]
                    ])

    w_r = calculate_absolute_gain(target_r, reference_r)
    w_g = calculate_absolute_gain(target_g, reference_g)
    w_b = calculate_absolute_gain(target_b, reference_b)
    weight_values =  np.array([w_r, w_g, w_b])
    return weight_values

def calculate_weights_ccm(file_path, validate=False):
    rgb_float32 = file_path
    color_checker_patches = detect_colour_checkers_segmentation(rgb_float32, show=False)
    if not color_checker_patches:
        print(f'WARNING! failed detection, skipping to next image')
        return None

    wb_weights = calculate_wb_weights_llsr(color_checker_patches[0]) #LLSR weights
    color_checker_white_balanced = color_checker_patches[0] * wb_weights

    color_checker_reference_values = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    m_t = color_checker_white_balanced
    print(f'M_T = {m_t}')
    m_r = np.zeros([24,3], dtype=np.float32) # ground truth values
    for index, (name, val) in enumerate(color_checker_reference_values.data.items()):
        xyz_space = xyY_to_XYZ(val)
        rgb_linear = XYZ_to_RGB(xyz_space, colourspace='sRGB',
                                       illuminant=color_checker_reference_values.illuminant,
                                       apply_cctf_encoding=False, dtype=np.float32)

        m_r[index] = rgb_linear

    ccm = matrix_colour_correction(m_t, m_r, method='Finlayson 2015')

    if validate:
        delta_e = validate_color_correction(ccm, m_t, m_r)
        for i, val in enumerate(delta_e):
            print(f'Patch {i + 1:02d} Delta E = {val:.4f}')

    return wb_weights, ccm


def apply_weights_ccm(image_path, weights, ccm):

    # Read raw .CR2 image, convert to linear float32
    rgb_float32 = raw_to_rgb(image_path)

    # Apply White balancing
    rgb_float32 *= weights
    print(f'Weights = {weights}')

    # Apply Color Correction Matrix (CCM)
    rgb_color_corrected = apply_matrix_colour_correction(rgb_float32, ccm, method='Finlayson 2015')

    # Apply Gamma correction
    rgb_gamma_corrected = cctf_encoding(rgb_color_corrected)

    # Clip image 0.0 - 1.0, cast to uint8 for export
    srgb_uint8 = np.round(np.clip(rgb_gamma_corrected, 0.0, 1.0) * 255).astype(np.uint8)
    return srgb_uint8

def validate_paths(parent_directory):
    # parent_directory should be a PathLib Path object.
    v_paths = []
    for path in parent_directory.iterdir():
        if path.is_dir():
            source_path = path / 'images_source'.lower()
            color_checker = source_path / 'calibration.cr2'.lower()
            if color_checker.is_file():
                v_paths.append(path)
            else:
                print(f'WARNING! No color checker found in {source_path}')
    return sorted(v_paths)

def process_images(image_paths):
    for path in image_paths:
        source_path = path / 'images_source'
        output_path = path / 'images_calibrated'
        output_path.mkdir(exist_ok=True)
        c_checker_path = source_path/'calibration.cr2'.lower()
        weights_ccm = calculate_weights_ccm(raw_to_rgb(c_checker_path), validate=True)
        if weights_ccm is None:
            continue
        w = weights_ccm[0]
        ccm = weights_ccm[1]

        for f in source_path.iterdir():
            if f.suffix.lower() == ".cr2":
                print(f'Processing {f}')
                file_path = Path(output_path / f.name).with_suffix('.tga')
                processed_file = apply_weights_ccm(f, w, ccm)
                if processed_file is not None:
                    print(f'Saving {file_path}')
                    iio.imwrite(str(file_path), processed_file)

def validate_color_correction(ccm, target, reference):
    # Computer Delta E values per patch
    srgb = RGB_COLOURSPACES['sRGB']
    cc_target= apply_matrix_colour_correction(target, ccm, method='Finlayson 2015')
    target_xyz = RGB_to_XYZ(cc_target, colourspace='sRGB')
    reference_xyz = RGB_to_XYZ(reference, colourspace='sRGB')
    target_lab = XYZ_to_Lab(target_xyz, illuminant=srgb.whitepoint)
    reference_lab = XYZ_to_Lab(reference_xyz, illuminant=srgb.whitepoint)

    delta_e_values = delta_E(target_lab, reference_lab, method='CIE 2000')
    return delta_e_values



class Window(QMainWindow):
    def __init__(self):
        super().__init__()



if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6)
    p_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')

    valid_paths = validate_paths(p_directory)

    process_images(valid_paths)







