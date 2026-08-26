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
                               QFileDialog, QVBoxLayout, QHBoxLayout, QBoxLayout, QWidget, QStyle, QTableWidget,
                               QTableWidgetItem, QComboBox)


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
        return None

    wb_weights = calculate_wb_weights_llsr(color_checker_patches[0]) #LLSR weights
    color_checker_white_balanced = color_checker_patches[0] * wb_weights

    color_checker_reference_values = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    m_t = color_checker_white_balanced
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

    # Apply Color Correction Matrix (CCM)
    rgb_color_corrected = apply_matrix_colour_correction(rgb_float32, ccm, method='Finlayson 2015')

    # Apply Gamma correction
    rgb_gamma_corrected = cctf_encoding(rgb_color_corrected)

    # Clip image 0.0 - 1.0, cast to uint8 for export
    srgb_uint8 = np.round(np.clip(rgb_gamma_corrected, 0.0, 1.0) * 255).astype(np.uint8)
    return srgb_uint8

def validate_paths(parent_directory, log_callback=None):
    # parent_directory should be a PathLib Path object.
    v_paths = []
    for path in parent_directory.iterdir():
        if path.is_dir():
            source_path = path / 'images_source'.lower()
            color_checker = source_path / 'calibration.cr2'.lower()
            if source_path.is_dir() and color_checker.is_file():
                v_paths.append(path)
            else:
                log_callback(f"WARNING! No 'images_source' folder or color checker found in {source_path}")
    return sorted(v_paths)

def process_images(image_paths, export_format = '.tga', log_callback=None):
    for path in image_paths:
        log_callback(f'\nProcessing subfolder: {path}')
        source_path = path / 'images_source'
        output_path = path / 'images_calibrated'
        c_checker_path = source_path/'calibration.cr2'.lower()

        weights_ccm = calculate_weights_ccm(raw_to_rgb(c_checker_path), validate=False)
        if weights_ccm is None:
            log_callback(f'WARNING! Patch detection failed on {c_checker_path}, skipping to next folder')
            continue
        else:
            log_callback(f'Creating output path {output_path}')
            output_path.mkdir(exist_ok=True)

        w = weights_ccm[0]
        ccm = weights_ccm[1]

        for f in source_path.iterdir():
            if f.suffix.lower() == ".cr2":
                log_callback(f'\nProcessing {f}...')
                file_path = Path(output_path / f.name).with_suffix(export_format)
                processed_file = apply_weights_ccm(f, w, ccm)
                if processed_file is not None:
                    log_callback(f'Saving {file_path}...')
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
        self.parent_folder = "Choose Folder..."
        self.valid_paths = []

        # Set the window title and initial size
        self.setWindowTitle("Color Checker Batch Color Correction")
        self.resize(1200, 600)
        self.setStyleSheet("background-color: lightgrey;")

        # Create layout and attributes
        VLayout_main = QVBoxLayout()
        VLayout_main.setDirection(QBoxLayout.Direction.TopToBottom)
        VLayout_main.setSpacing(10)

        HLayout1 = QHBoxLayout()
        HLayout1.setDirection(QBoxLayout.Direction.LeftToRight)

        HLayout2 = QHBoxLayout()
        HLayout2.setDirection(QBoxLayout.Direction.LeftToRight)

        VLayout1 = QVBoxLayout()
        VLayout1.setDirection(QBoxLayout.Direction.TopToBottom)


        # Create button widgets
        self.folder_button = QPushButton()
        self.folder_button.setFixedSize(32, 32)
        self.folder_button.setStyleSheet("background-color: transparent; border: none;")
        folder_button_style = self.folder_button.style()
        folder_icon = folder_button_style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self.folder_button.setIcon(folder_icon)
        self.folder_button.setIconSize(QSize(32, 32))
        self.folder_button.clicked.connect(self.get_parent_folder)


        self.search_button = QPushButton("Scan Parent Folder")
        self.search_button.setEnabled(False) #Keep disabled until initial file selected
        self.search_button.setFixedSize(200,50)
        self.search_button.clicked.connect(self.scan_folders)

        self.process_button = QPushButton("Process Images")
        self.process_button.setEnabled(False) #Keep disabled until valid directories found
        self.process_button.setFixedSize(200,50)
        self.process_button.clicked.connect(self.process_images)



        # Create file path text widgets
        self.display_parent_folder = QLineEdit(self.parent_folder)
        self.display_parent_folder.setReadOnly(True)
        self.display_parent_folder.setStyleSheet("background-color: white; border: none;")

        # Create Label Widgets
        self.label1 = QLabel("Parent Folder: ", self)
        self.label2 = QLabel("Search Directory: ", self)
        self.Dropdown_label = QLabel('Choose Export Format:')


        # #Creat Table Widget
        # self.table = QTableWidget()
        # self.table_rows = 10
        # self.table_columns = 3
        # self.table.setColumnCount(self.table_columns)
        # self.table.setRowCount(self.table_rows)
        # self.table.setStyleSheet("background-color: white; border: none;")
        # self.table.setItem(2, 1, QTableWidgetItem("Parent Folder: "))

        #Create a Combo Box (drop down menu) Widget

        self.dropdown = QComboBox()
        self.dropdown.setFixedSize(200, 25)
        # self.dropdown.setStyleSheet("background-color: white;")
        self.dropdown.insertItems(0, ['.tga', '.png', '.tif', '.jpg', '.bmp'])
        self.dropdown.currentIndexChanged.connect(self.log_dropdown)

        # Create Terminal Widget
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(100)
        self.terminal.appendPlainText('Log Updates...')
        self.terminal.setMinimumHeight(250)
        self.terminal.setStyleSheet("background-color: white;")



        # Assemble the layouts, add widgets, nest HLayouts under VLayout_main
        HLayout1.addWidget(self.label1)
        HLayout1.addWidget(self.display_parent_folder)
        HLayout1.addWidget(self.folder_button)

        VLayout1.setSpacing(20)
        VLayout1.addWidget(self.search_button)
        VLayout1.addWidget(self.process_button)
        VLayout1.addWidget(self.Dropdown_label)
        VLayout1.addWidget(self.dropdown)

        VLayout1.addStretch()



        HLayout2.addLayout(VLayout1)
        HLayout2.addWidget(self.terminal)


        # Main assembly
        VLayout_main.addLayout(HLayout1)
        VLayout_main.addLayout(HLayout2)


      # Create a central widget, set the layout, and apply it to the QMainWindow
        central_widget = QWidget()
        central_widget.setLayout(VLayout_main)
        self.setCentralWidget(central_widget)

    def get_parent_folder(self):
        selected_dir = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select Directory",
            dir="",  # Starting directory (empty string defaults to the current working directory)
            options=QFileDialog.Option.ShowDirsOnly  # Restricts selection to directories only
        )
        if not selected_dir:
            self.terminal.appendPlainText('Selection cancelled')
            return
        self.parent_folder = Path(selected_dir)
        self.display_parent_folder.setText(str(self.parent_folder))
        self.terminal.appendPlainText(f'Parent folder: {str(self.parent_folder)}')
        self.search_button.setEnabled(True)

    def scan_folders(self):
        self.search_button.setEnabled(False)
        self.terminal.appendPlainText('\nScanning for valid subfolders...')
        self.valid_paths = validate_paths(self.parent_folder, self.terminal.appendPlainText)
        if len(self.valid_paths) == 0:
            self.terminal.appendPlainText('No valid subfolders found')
            self.process_button.setEnabled(False)
        else:
            self.terminal.appendPlainText('\nValid paths found:')
            self.process_button.setEnabled(True)
            for v in self.valid_paths:
                self.terminal.appendPlainText(str(v))

        self.search_button.setEnabled(True)

    def process_images(self):
        print("Background task started...")
        self.search_button.setEnabled(False)
        self.terminal.appendPlainText('\nProcessing images...')
        process_images(self.valid_paths,self.dropdown.currentText(), self.terminal.appendPlainText)
        self.search_button.setEnabled(True)
        print("Background task finished!")

    def log_dropdown(self):
        self.terminal.appendPlainText(f'Export format changed to: {self.dropdown.currentText()}')



if __name__ == "__main__":

    # Create the application instance
    app = QApplication(sys.argv)

    # Create and show the window
    window = Window()
    window.show()

    # Start the event loop
    sys.exit(app.exec())









