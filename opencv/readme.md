# Instructions

To set up and run the benchmark you need to do the following

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 benchmark.py
```

SURF is not included in the pip version of OpenCV. You
will need to build OpenCV 4.x from source code to get it. However, building
building from source code breaks the benchmark rule of using the most common
and easiest way to get the library running. FYI some operations,
such as Guassian blur, run 2x faster and while others are exactly the
same, e.g. canny edge. I think some algorithms contain hand 
optimized code for certain architectures.


Example Output:

```text
Gaussian Blur   12.1 ms
meanThresh      15.4 ms
gradient sobel  14.0 ms
histogram       7.0 ms
canny           61.7 ms
Skipping SIFT and SURF. Not installed
contour         83.3 ms
good features   277.2 ms
hough polar     3086.3 ms
```


