import rawpy, colour
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour import CCS_COLOURCHECKERS, matrix_colour_correction, apply_matrix_colour_correction
from colour_checker_detection import detect_colour_checkers_segmentation

def raw_to_rgb(image_path):
    # Convert raw .CR2 files to a float32 linear sRGB NumPy Array
    with rawpy.imread(str(image_path)) as raw:
        rgb_uint16 = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,

                # 1. STOP RAWPY FROM WHITE BALANCING (Gemini Pro)
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],  # Forces a 1.0 multiplier across R, G, B, and G2

                # 2. STOP RAWPY FROM APPLYING A BUILT-IN COLOR MATRIX (Gemini Pro)
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

    neutral_grey = color_checker_patches[0][21] #patch 22 is ~18% grey

    # calculate rgb weight values
    epsilon = 0.00001
    neutral_reflectance = 0.191 #Color Checker Classic baseline post-2014 pigment changes
    weights = neutral_reflectance / np.maximum(neutral_grey, epsilon)

    color_checker_white_balanced = color_checker_patches[0] * weights
    color_checker_target_values = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    M_R = color_checker_white_balanced
    M_T = np.zeros([24,3], dtype=np.float32)
    for index, (name, val) in enumerate(color_checker_target_values.data.items()):
        xyz_space = colour.xyY_to_XYZ(val)
        rgb_linear = colour.XYZ_to_RGB(xyz_space, colourspace='sRGB',
                                       illuminant=color_checker_target_values.illuminant,
                                       apply_cctf_encoding=False, dtype=np.float32)

        M_T[index] = rgb_linear
    ccm = matrix_colour_correction(M_T, M_R)
    return weights, ccm

def apply_weights_ccm(image_path, weights, ccm):
    rgb_float32 = raw_to_rgb(image_path)
    rgb_float32 *= weights

    # Apply Color and gamma correction
    rgb_color_corrected = apply_matrix_colour_correction(rgb_float32, ccm)
    rgb_gamma_corrected = colour.cctf_encoding(rgb_color_corrected)

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






np.set_printoptions(suppress=True, precision=6)
p_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')

valid_paths = validate_paths(p_directory)
process_images(valid_paths)





