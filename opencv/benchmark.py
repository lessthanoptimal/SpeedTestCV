import cv2
import statistics
import time
import numpy as np

num_trials = 10
image_path = "../data/chessboard_large.jpg"
img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

region_radius = 5
region_width = region_radius*2+1

def gaussianBlur():
    cv2.GaussianBlur(img, (region_width, region_width), 0)

def gradientSobel():
    cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

def meanThresh():
    cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, region_width, 0)

def goodFeatures():
    # For comparing the two algorithms quality of zero would be idea because chances are the meaning
    # of quality is off by a scale factor. OpenCV doesn't allow that so a small value is used instead
    cv2.goodFeaturesToTrack(img,1000,qualityLevel=0.000001,minDistance=2)


# TODO SURF

# TODO SIFT

# TODO contour
def contour():
    # Code in Validation Boof attempted to replicate the same behavior in both libraries. Configuration
    # values are taken from there.
    # The algorithm in OpenCV and thw two contour algorithms in BoofCV operate internally very different
    pass

# TODO Canny - Skipped due to difficulty in configuring them the same way

def houghLine():
    # OpenCV contains 3 Hough Line algorithms. CV_HOUGH_STANDARD is the closest to the variants included
    # in boofcv. The other OpenCV variants should have very different behavior based on their description
    cv2.HoughLines(img, rho=1, theta=np.pi/180, threshold=100)

def benchmark( f ):
    times=[]
    for trials in range(num_trials):
        t0 = time.time()
        f()
        t1 = time.time()
        times.append(t1-t0)
    return statistics.mean(times)*1000

# print("Gaussian Blur   {:.1f} ms".format(benchmark(gaussianBlur)))
# print("meanThresh      {:.1f} ms".format(benchmark(meanThresh)))
# print("gradient sobel  {:.1f} ms".format(benchmark(gradientSobel)))
# print("good features   {:.1f} ms".format(benchmark(goodFeatures)))
print("hough polar     {:.1f} ms".format(benchmark(houghLine)))

print()
print("Done!")