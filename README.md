Color Correction batch processor leveraging 'Color Checker Classic 24 After 2014'.

1. Scans a parent folder for subfolders that contain an 'images_source' directory.
2. Scans 'images_source' subfolder for a valid 'calibration.CR2' file and attempts to auto-detect color checker patches.
3. If successful, will pull patch values, calculate white balance weights using Linear Least Square Regression on all 6 neutral patches
4. Calculate Color Correction Matrix(CCM) based off the White balanced patches.
5. Apply white balance and CCM to all .cr2 files, write out gamma corrected sRGB uint8 files to an 'image_calibrated' subfolder.

Useful for color correcting large amounts of photographs for photogrammetry pipelines, etc. I have also included a Delta E function
for calculating accuracy of each patch relative to the 'Color Checker Classic 24 After 2014' reference values

Steps:
1. Create a parent folder.
2. Create subfolders for each photo session making sure you capture one picture of the Color Checker per session.
3. Rename the Color Checker photo to 'calibration.cr2'.
4. Place all of the .cr2 files in a new subfolder named 'images_source' under each session folder. Example: Photogrammetry/Rock_A/images_source/, Photogrammetry/Rock_B/images_source/, Photogrammetry/Rock_C/images_source/, etc.
5. Point to the parent folder and run the script. It will scan all subfolders to validate, auto-detect the calibration.cr2 patch values, and output calibrated pictures.

   
