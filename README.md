# Webb Deep Field Galaxy Detector

A Python program that loads real James Webb Space Telescope imaging data 
and automatically detects and maps bright astronomical objects using 
image processing and object detection techniques.

## What It Does

- Loads real Webb First Deep Field telescope imagery
- Converts the image into numerical data for analysis
- Analyzes brightness levels across nearly 1 million pixels
- Automatically identifies bright astronomical objects
- Draws detection markers around each identified object
- Detected 235 bright objects in a single deep field image

## Why This Is Interesting

The Webb First Deep Field image contains thousands of galaxies in a patch 
of sky the size of a grain of sand held at arm's length. Each bright object 
detected by this program is a galaxy containing hundreds of billions of stars. 
The light from some of these galaxies left before Earth existed.

This project applies the same anomaly detection concepts from my previous 
cybersecurity simulation to real astronomical data — demonstrating that 
machine learning techniques transfer across completely different domains.

## Technical Details

- Language: Python 3.11
- Libraries: PIL, NumPy, Matplotlib, SciPy
- Technique: Brightness thresholding and connected component labeling
- Data source: James Webb Space Telescope First Deep Field (NASA)

## Results

- Image analyzed: 1280 x 720 pixels (921,600 total pixels)
- Average brightness: 30.84 (image is mostly deep space)
- Bright objects detected: 235
- Bright pixels (>200 brightness): 2,971 (0.3% of image)

## Connection to Previous Work

This project extends my previous work on ML-based anomaly detection 
in cyber-physical systems. Instead of detecting sensor anomalies in a 
robot simulation, this program detects brightness anomalies in real 
space telescope data. The underlying detection logic is identical — 
find what does not match the baseline and flag it.

## Author

Atharv Kumaria — 7th grade, Sammamish WA  
Built independently as part of ongoing self-directed CS research and space science research.
