Compares the speed of common image processing routines in different computer vision libraries. The goal is to emulate the performance of what the average user can be expected to experience. Essentially if you spent only a few minutes following the beginner tutorial what sort of results would you expect. Correctness of an implementation is not considered. If it is known that an implementation is flat out wrong then it will be omitted with the reason stated.

Functions Compared:
* Gaussian Blur
* Sobel Gradient
* Local Mean Threshold
* Image Histogram
* Canny Edge
* Binary Contour (external)
* Corner Features (Shi-Tomasi)
* Hough Line (polar)
* SIFT Detect and Describe
* SURF Detect and Describe

Neural networks are absent since they require a GPU to be of practical use.

Rules:
1) Pre-built binaries from official sources only
    * No custom builds from source.
    * No external addons
2) Stick to techniques shown in beginner examples and tutorials
3) Each function is tuned to produce similar results across libraries
    * If possible this is checked by rendering the output.
4) CPU code only, no GPU or other special hardware specific code
5) Concurrent code/threads are allowed, but only when transparently applied by the library
6) Virtual machine environments are allowed to have a warm up
7) Use proper micro-benchmarking techniques whenever possible

## Building and Running

See readme.md inside each libraries' directory for instructions.

## Tuning

Tuning the algorithms so that they produce similar results is difficult to impossible. For algorithms where the speed
depends significantly on the number of features found the target features as selected to be what looked like a 
reasonable number of an image of this size. If the output from the implementation isn't in a usable format, then additional
processing was done. Examples are Canny in OpenCV where it outputs a binary image but you need a
list of pixels in the contour and in BoofCV the contour is computed in a compact format, which is then decompressed 
into a pixel format.

* Gaussian Blur: radius = 5
* Local Mean Threshold: radius = 5
* Canny: Output edge pixel chains. 550,000 unique pixels in chains expected.
* Contour: External contours only. 4-connect rule. Should find around 1,111,793 points
* Corners: Shi-Tomasi. Unweighted variant. radius=21. 3000 features
* Hough Line Polar: resolutions( angle=1 deg, range= 5 pix) tune to detect 500 lines
* SIFT: 5 octaves, 10,000 features
* SURF: 4 octaves, 4 scales, 10,000 features

## Comparing Implementations

Creating a fair comparison between libraries is extremely difficult. It's extremely rare that any two libraries implement the same algorithm the exact same way. When two implementations have the same tuning parameters the meaning of a tuning parameter's value is like to be different, e.g. off by a scale factor or absolute vs relative.

Even for very simple and well defined tasks like convolving an image or converting an RGB image to gray scale there are issues. Here are now some examples. When applying Gaussian blur to an integer images you need to round the results. As far as I know, every library but BoofCV does this by applying regular integer division. BoofCV does it by rounding instead, making it more mathematically correct. There is no universally accepted way to convert an RGB image into a Gray image. OpenCV performs a weighted average. BoofCV by default does a regular average but as the option to do a similar weighted average. The situation gets worse exponentially as the complexity of each algorithm increases.

This benchmark attempts to address that issue by not requiring the exact same output but instead trying to produce similar results. If trying to detect lines, don't require that all the lines be the same just most of them and detect about the same number. Each operation in each library is individually tuned and results are in some way visualized and compared.