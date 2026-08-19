import rawpy, colour
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_checker_detection import detect_colour_checkers_segmentation


def calculate_weights(color_checker_path):
    # Images should be raw / linear .cr2 files
    with rawpy.imread(str(color_checker_path)) as raw:
        rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values
        color_checker_patches = detect_colour_checkers_segmentation(rgb_float32, show=False)
        if not color_checker_patches:
            print(f'WARNING! {color_checker_path} failed detection, skipping to next image')
            return
        neutral_grey = color_checker_patches[0][21] #patch 22 is 18% grey
        print(len(neutral_grey))

    # calculate rgb weight values
    EPSILON = 0.00001
    weights = 0.18 / np.maximum(neutral_grey, EPSILON)
    return weights


def white_balance(image_source_path, weights):
    # images should be raw / linear .cr2 files
    with rawpy.imread(str(image_source_path)) as raw:
        rgb_uint16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # uint16 numpy array [x,y,[r,g,b]]
        rgb_float32 = rgb_uint16.astype(np.float32) / 65535.0  # cast to float32, normalize rgb values

    # Apply weights and perform sRGB gamma correction
    rgb_float32 *= weights
    srgb_float32 = colour.cctf_encoding(rgb_float32)

    # Clip image 0.0 - 1.0, cast to uint8 for export
    srgb_uint8 = np.round(np.clip(srgb_float32, 0.0, 1.0) * 255).astype(np.uint8)
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
    return sorted(v_paths)

def process_images(image_paths):
    for path in image_paths:
        Path.mkdir(path / 'images_calibrated'.lower(), exist_ok=True)
        source_path = Path(path / 'images_source')
        output_path = Path(path / 'images_calibrated')
        c_checker_path = Path(source_path/'calibration.cr2'.lower())
        w = calculate_weights(c_checker_path)
        for f in source_path.iterdir():
            file_path = Path(output_path / f.name).with_suffix('.tga')
            print(f'writing file:  {file_path}')
            # iio.imwrite(str(file_path), white_balance(f, w))







np.set_printoptions(suppress=True, precision=6)
parent_directory = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry')

valid_paths = validate_paths(parent_directory) #get valid subfolder list, create export folders
process_images(valid_paths) #export corrected images



# export_path = 'images_calibrated'
# for paths in valid_paths:
#     for file in paths.iterdir():
#         if file.is_file() and file.suffix.lower() == '.cr2':
#             print(file)
#
#
# cr2_test_source_path = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry/lighting_orange/orange_light.CR2')
# cr2_test_export_path = Path('/Users/renderman/Documents/Python/Image_processing/Photogrammetry/cliff_a/test_exprt.tiff')
#
# image = rawpy.imread(str(cr2_test_source_path)) #cast to string since rawpy wont accept pathlib objects
# white_balance(cr2_test_source_path)
#
# iio.imwrite(str('/Users/renderman/Documents/Python/Image_processing/test_exprort_sRGB.tga'), white_balance(cr2_test_source_path))







