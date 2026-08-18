import rawpy, colour
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_checker_detection import detect_colour_checkers_segmentation




def white_balance(image_source_path):
    # images should be raw / linear .cr2 files
    with rawpy.imread(str(image_source_path)) as raw:
        rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values
        neutral_grey = detect_colour_checkers_segmentation(rgb_float32, show=True)[0][21] # Colorchecker Classic Patch 22

    # calculate rgb weight values
    EPSILON = 0.00001
    weights = 0.18 / np.maximum(neutral_grey, EPSILON)

    # Apply weights and perform sRGB gamma correction
    rgb_float32 *= weights
    srgb_float32 = colour.cctf_encoding(rgb_float32)

    # Clip image 0.0 - 1.0, cast to uint8 for export
    srgb_uint8 = np.round(np.clip(srgb_float32, 0.0, 1.0) * 255).astype(np.uint8)
    return srgb_uint8


def valid_paths(parent_directory):
    # parent_directory should be a PathLib Path object. Search parent folder for valid subfolders.
    # return pathlib object list if color checker present, create output subfolder
    v_paths = []
    for path in parent_directory.iterdir():
        if path.is_dir():
            source_path = path / 'images_source'.lower()
            color_checker = source_path / 'calibration.cr2'.lower()
            if color_checker.is_file():
                v_paths.append(source_path)
                print(source_path)
                output_path = path / 'images_calibrated'.lower()
                # Path.mkdir(output_path, exist_ok=True)
    return sorted(v_paths)








np.set_printoptions(suppress=True, precision=6)
parent_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')
valid_paths = valid_paths(parent_directory)


export_path = 'images_calibrated'
for paths in valid_paths:
    for file in paths.iterdir():
        if file.is_file() and file.suffix.lower() == '.cr2':
            print(file)


cr2_test_source_path = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry/lighting_orange/orange_light.CR2')
cr2_test_export_path = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry/cliff_a/test_exprt.tiff')

image = rawpy.imread(str(cr2_test_source_path)) #cast to string since rawpy wont accept pathlib objects
white_balance(cr2_test_source_path)

iio.imwrite(str('/Users/renderman/Documents/Python/Image_processing/test_exprort_sRGB.tga'), white_balance(cr2_test_source_path))







