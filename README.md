Color Correction batch processor leveraging 'Color Checker Classic 24 After 2014'.

1. Scans a parent folder for subfolders that contain an 'images_source' directory.
2. Scans 'images_source' subfolder for a valid 'calibration.CR2' file and attempts to auto-detect color checker patches.
3. If successful, will pull patch values, calculate white balance weights using Linear Least Square Regression on all 6 neutral patches
4. Calculate Color Correction Matrix(CCM) based off the White balanced patches.
5. Apply white balance and CCM to all .cr2 files, write out gamma corrected sRGB uint8 files to an 'image_calibrated' subfolder.

Useful for color correcting large amounts of photographs for photogrammetry pipelines, etc. I have also included a Delta E function
for calculating accuracy of each patch relative to the 'Color Checker Classic 24 After 2014' reference values

   
