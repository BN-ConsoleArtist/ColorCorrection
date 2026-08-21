import rawpy, colour
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour import CCS_COLOURCHECKERS, matrix_colour_correction
from colour_checker_detection import detect_colour_checkers_segmentation

def calculate_ccm(color_checker_path):
    my_checker = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

    # Images should be raw / linear .cr2 files
    with rawpy.imread(str(color_checker_path)) as raw:
        rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values

    color_checker_patches = detect_colour_checkers_segmentation(rgb_float32, show=False)
    if not color_checker_patches:
        print(f'WARNING! {color_checker_path} failed detection, skipping to next image')
        return None
    neutral_grey = color_checker_patches[0][21] # neutral 5 (patch 22)

    # calculate rgb weight values
    epsilon = 0.00001
    neutral_reflectance = 0.191  # Color Checker Classic baseline post-2014 neutral 5 patch
    weights = neutral_reflectance / np.maximum(neutral_grey, epsilon)

    # White balance
    color_checker_white_balanced = color_checker_patches[0] * weights #M_T

    M_T = color_checker_white_balanced
    M_R = np.zeros([24,3], dtype=np.float32)
    # print(M_T.shape)
    # print(M_R.shape)
    for index, (name, val) in enumerate(my_checker.data.items()):
        xyz_space = colour.xyY_to_XYZ(val)
        rgb_linear = colour.XYZ_to_RGB(xyz_space, colourspace='sRGB', illuminant=my_checker.illuminant, apply_cctf_encoding=False, dtype=np.float32)
        M_R[index] = rgb_linear

    ccm = matrix_colour_correction(M_T, M_R)
    # print(f'Neutral 5 = {M_T[21]}')
    # print(f'CCM = {ccm}')
    return ccm

def apply_ccm(image_path, ccm):
    if ccm == None:
        return None

    with rawpy.imread(str(image_path)) as raw:
        rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values

    corrected_image_rgb_float32 = colour.apply_matrix_colour_correction(rgb_float32, ccm)

    # apply gamme correction, clamp values, cast to unint8
    corrected_image_cctf_float32 = colour.cctf_encoding(corrected_image_rgb_float32)
    corrected_image_srgb_uint8 = np.round(np.clip(corrected_image_cctf_float32, 0.0, 1.0) * 255).astype(np.uint8)
    return corrected_image_srgb_uint8








# def calculate_weights(color_checker_path):
#     # Images should be raw / linear .cr2 files
#     with rawpy.imread(str(color_checker_path)) as raw:
#         rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
#         rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values
#
#         color_checker_patches = detect_colour_checkers_segmentation(rgb_float32, show=False)
#         if not color_checker_patches:
#             print(f'WARNING! {color_checker_path} failed detection, skipping to next image')
#             return None
#
#         neutral_grey = color_checker_patches[0][21] #patch 22 is ~18% grey
#
#     # calculate rgb weight values
#     epsilon = 0.00001
#     neutral_reflectance = 0.191 #Color Checker Classic baseline post-2014 pigment changes
#     weights = neutral_reflectance / np.maximum(neutral_grey, epsilon)
#     return weights
#
#
# def white_balance(image_source_path, weights):
#     # images should be raw / linear .cr2 files
#     with rawpy.imread(str(image_source_path)) as raw:
#         rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
#         rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values
#
#     # Apply weights and perform sRGB gamma correction
#     rgb_float32 *= weights
#     srgb_float32 = colour.cctf_encoding(rgb_float32)
#
#     # Clip image 0.0 - 1.0, cast to uint8 for export
#     srgb_uint8 = np.round(np.clip(srgb_float32, 0.0, 1.0) * 255).astype(np.uint8)
#     return srgb_uint8
#
# def color_correction(rgb_float32):
#     my_checker = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
#     M_T = rgb_float32
#     M_R = np.zeroes([24,3], dtype=np.float32)
#     for index, (name, val) in enumerate(my_checker.data.items()):
#         XYZ_Space = colour.xyY_to_XYZ(val)
#         RGB_Linear = colour.XYZ_to_RGB(XYZ_Space, colourspace='sRGB', illuminant=my_checker.illuminant, apply_cctf_encoding=False, dtype=np.float32)
#         M_R[index] = RGB_Linear
#         # print(RGB_Linear)
#
#     ccm = matrix_colour_correction(M_T, M_R)
#
#     return None


def validate_paths(parent_directory):
    # parent_directory should be a PathLib Path object.
    v_paths = []
    for path in parent_directory.iterdir():
        if path.is_dir():
            source_path = path / 'images_source'.lower()
            color_checker = source_path / 'calibration.cr2'.lower()
            if color_checker.is_file():
                v_paths.append(path)
    return sorted(v_paths)

# def validate_results(parent_directory):
#     for path in parent_directory.iterdir():
#         if path.is_dir():
#             # print(path)
#             for p in path.iterdir():
#                 if p.is_dir() and p.name == 'images_calibrated':
#                     # print(p)
#                     for f in p.iterdir():
#                         if f.is_file() and f.name == 'calibration.tga': #files are sRGB uint8, RGB should be [119,119,119]
#                             print(f)
#                             file = iio.imread(str(f))
#                             color_checker_patches = detect_colour_checkers_segmentation(file, show=False)
#                             if color_checker_patches:
#                                 print(f'Neutral Grey = {np.round(color_checker_patches[0][21] * 255.0)}')




def process_images(image_paths):
    for path in image_paths:
        source_path = path / 'images_source'
        output_path = path / 'images_calibrated'
        output_path.mkdir(exist_ok=True)
        c_checker_path = source_path/'calibration.cr2'.lower()

        # w = calculate_weights(c_checker_path)
        # if w is None:
        #     continue

        # for f in source_path.iterdir():
        #     file_path = Path(output_path / f.name).with_suffix('.tga')
        #     print(f'writing file:  {file_path}')
        #     iio.imwrite(str(file_path), white_balance(f, w))

        color_correction_matrix = calculate_ccm(c_checker_path)

        for f in source_path.iterdir():
            print(f)
            print(apply_ccm(f, color_correction_matrix))
        #     file_path = Path(output_path / f.name).with_suffix('.tga')
        #     print(f'writing file:  {file_path}')
        #     iio.imwrite(str(file_path), apply_ccm(f, color_correction_matrix))






np.set_printoptions(suppress=True, precision=6)
p_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')

valid_paths = validate_paths(p_directory)
process_images(valid_paths)
# cc_path = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry/lighting_blue/images_source/calibration.CR2')
# calculate_ccm(cc_path)
#
# valid_paths = validate_paths(p_directory) #get valid subfolder list, create export folders
# process_images(valid_paths) #export corrected images
#
# validate_results(p_directory)




