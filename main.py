import rawpy
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_checker_detection import detect_colour_checkers_segmentation
from colour import (CCS_COLOURCHECKERS, matrix_colour_correction, apply_matrix_colour_correction,
                    xyY_to_XYZ, XYZ_to_RGB, cctf_encoding)
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

def calculate_weights_ccm(linear_rgb_image_array):
    rgb_float32 = linear_rgb_image_array
    color_checker_patches = detect_colour_checkers_segmentation(rgb_float32, show=False)
    if not color_checker_patches:
        print(f'WARNING! failed detection, skipping to next image')
        return None

    # neutral_grey = color_checker_patches[0][21] #patch 22 is ~18% grey
    neutral_patches = color_checker_patches[0][18:24] #Gemini suggestion
    neutral_grey_average = np.mean(neutral_patches, axis=0) #Gemini suggestion

    # calculate rgb weight values
    epsilon = 0.00001
    neutral_reflectance = 0.191 #Color Checker Classic baseline post-2014 pigment changes
    # weights = neutral_reflectance / np.maximum(neutral_grey, epsilon)
    weights = neutral_reflectance / np.maximum(neutral_grey_average, epsilon) #Gemini suggestion

    color_checker_white_balanced = color_checker_patches[0] * weights
    color_checker_target_values = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    m_t = color_checker_white_balanced
    m_r = np.zeros([24,3], dtype=np.float32) # ground truth values
    for index, (name, val) in enumerate(color_checker_target_values.data.items()):
        xyz_space = xyY_to_XYZ(val)
        rgb_linear = XYZ_to_RGB(xyz_space, colourspace='sRGB',
                                       illuminant=color_checker_target_values.illuminant,
                                       apply_cctf_encoding=False, dtype=np.float32)

        m_r[index] = rgb_linear
    # ccm = matrix_colour_correction(m_t, m_r)
    ccm = calculate_exposure_constrained_ccm(m_t, m_r) # Gemini suggestion

    return weights, ccm


def calculate_exposure_constrained_ccm(rgb_in, rgb_out, white_index=18):

    # This code was a suggestion from Google Gemini Pro
    """
    Calculates an optimal 3x3 CCM that perfectly preserves the neutral axis
    while automatically scaling exposure to match the ground truth.

    white_index: index of the white patch (default 18 for a standard 24-patch checker)
    """
    # 1. Find the exposure scaling factor (k)
    in_white = np.mean(rgb_in[white_index])
    out_white = np.mean(rgb_out[white_index])
    k = out_white / in_white

    # 2. Solve for the matrix with row sums strictly constrained to 'k'
    R_in, G_in, B_in = rgb_in[:, 0], rgb_in[:, 1], rgb_in[:, 2]

    # The math constraint: m1*(R - B) + m2*(G - B) = C_out - k*B
    A = np.column_stack((R_in - B_in, G_in - B_in))
    ccm = np.zeros((3, 3))

    for i in range(3):
        C_out = rgb_out[:, i]
        y = C_out - (k * B_in)

        # Calculate optimal m1 and m2
        x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        m1, m2 = x[0], x[1]
        m3 = k - m1 - m2  # Strictly enforce the exposure sum constraint

        # Store in rows (correctly formatted for the colour-science library)
        ccm[i] = [m1, m2, m3]

    return ccm

def apply_weights_ccm(image_path, weights, ccm):
    rgb_float32 = raw_to_rgb(image_path)
    rgb_float32 *= weights

    # Apply Color and gamma correction
    # rgb_color_corrected = apply_matrix_colour_correction(rgb_float32, ccm)
    rgb_color_corrected = apply_matrix_colour_correction(rgb_float32, ccm) # Gemini Pro Suggestion

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

        weights_ccm = calculate_weights_ccm(raw_to_rgb(c_checker_path))
        if weights_ccm is None:
            continue
        w = weights_ccm[0]
        ccm = weights_ccm[1]

        for f in source_path.iterdir():
            file_path = Path(output_path / f.name).with_suffix('.tga')
            processed_file = apply_weights_ccm(f, w, ccm)
            if processed_file is not None:
                print(f'Processing file in {file_path}')
                iio.imwrite(str(file_path), processed_file)

def validate_color_correction(v_paths):
    for path in v_paths:
        if path.is_dir():
            if Path(path / 'images_calibrated' / 'calibration.tga').is_file():
                file_path = path / 'images_calibrated' / 'calibration.tga'
                checker = iio.imread(str(file_path))
                checker = iio.imread(str(file_path)).astype(np.float32) / 255.0
                color_checker_patches = detect_colour_checkers_segmentation(checker, show=False)
                print(color_checker_patches[0] * 255)
    return None



class Window(QMainWindow):
    def __init__(self):
        super().__init__()



if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6)
    p_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')

    valid_paths = validate_paths(p_directory)
    process_images(valid_paths)
    validate_color_correction(valid_paths)






