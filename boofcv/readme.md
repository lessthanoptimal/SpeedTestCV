# Instructions

JMH is micro benchmark tool and is used to compute performance.

```bash
cd SpeedTestCV/boofcv
./gradlew jmh
```


Example Output:

```text
Benchmark                                 Mode  Cnt     Score     Error  Units
BenchmarkImageProcessing.canny            avgt    5   142.658 ±   2.893  ms/op
BenchmarkImageProcessing.contourExternal  avgt    5    47.055 ±   1.076  ms/op
BenchmarkImageProcessing.gaussianBlur     avgt    5    36.139 ±   4.140  ms/op
BenchmarkImageProcessing.goodFeatures     avgt    5    38.313 ±   2.730  ms/op
BenchmarkImageProcessing.histogram        avgt    5     2.268 ±   0.458  ms/op
BenchmarkImageProcessing.houghPolar       avgt    5   743.051 ±  26.191  ms/op
BenchmarkImageProcessing.sift             avgt    5  1387.928 ± 232.038  ms/op
BenchmarkImageProcessing.sobel            avgt    5     8.010 ±   0.941  ms/op
BenchmarkImageProcessing.surf             avgt    5   249.769 ±  49.476  ms/op
BenchmarkImageProcessing.threshMean       avgt    5    20.495 ±   9.822  ms/op
```

On my system, JDK 1.8 and JDK 11 produce similar output.