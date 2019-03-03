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

## Running The Benchmark

## Comparing Implementations

Creating a fair comparison between libraries is extremely difficult. It's extremely rare that any two libraries implement the same algorithm the exact same way. When two implementations have the same tuning parameters the meaning of a tuning parameter's value is like to be different, e.g. off by a scale factor or absolute vs relative.

Even for very simple and well defined tasks like convolving an image or converting an RGB image to gray scale there are issues. Here are now some examples. When applying Gaussian blur to an integer images you need to round the results. As far as I know, every library but BoofCV does this by applying regular integer division. BoofCV does it by rounding instead, making it more mathematically correct. There is no universally accepted way to convert an RGB image into a Gray image. OpenCV performs a weighted average. BoofCV by default does a regular average but as the option to do a similar weighted average. The situation gets worse exponentially as the complexity of each algorithm increases.

This benchmark attempts to address that issue by not requiring the exact same output but instead trying to produce similar results. If trying to detect lines, don't require that all the lines be the same just most of them and detect about the same number. Each operation in each library is individually tuned and results are in some way visualized and compared.