package speed;

import boofcv.abst.feature.detdesc.ConfigCompleteSift;
import boofcv.abst.feature.detdesc.DetectDescribePoint;
import boofcv.abst.feature.detect.interest.ConfigFastHessian;
import boofcv.abst.feature.detect.interest.ConfigGeneralDetector;
import boofcv.abst.feature.detect.line.DetectLine;
import boofcv.abst.filter.binary.BinaryContourFinder;
import boofcv.abst.filter.binary.InputToBinary;
import boofcv.abst.filter.blur.BlurFilter;
import boofcv.abst.filter.derivative.ImageGradient;
import boofcv.alg.feature.detect.edge.CannyEdge;
import boofcv.alg.feature.detect.interest.GeneralFeatureDetector;
import boofcv.alg.misc.ImageStatistics;
import boofcv.concurrency.BoofConcurrency;
import boofcv.core.image.ConvertImage;
import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
import boofcv.factory.feature.detect.edge.FactoryEdgeDetectors;
import boofcv.factory.feature.detect.interest.FactoryDetectPoint;
import boofcv.factory.feature.detect.line.ConfigHoughFoot;
import boofcv.factory.feature.detect.line.ConfigHoughPolar;
import boofcv.factory.feature.detect.line.FactoryDetectLineAlgs;
import boofcv.factory.filter.binary.FactoryBinaryContourFinder;
import boofcv.factory.filter.binary.FactoryBinaryImageOps;
import boofcv.factory.filter.binary.FactoryThresholdBinary;
import boofcv.factory.filter.blur.FactoryBlurFilter;
import boofcv.factory.filter.derivative.FactoryDerivative;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.ConfigLength;
import boofcv.struct.feature.BrightFeature;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayS16;
import boofcv.struct.image.GrayU8;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
@Fork(value=1)
public class BenchmarkImageProcessing {

//    @Param({"true","false"})
    public boolean concurrent=true;

    int radius = 5;
    ConfigLength kerLength = ConfigLength.fixed(radius*2+1);

    GrayU8 grayU8;
    GrayF32 grayF32;
    GrayU8 outputU8;
    GrayU8 binaryU8;

    GrayS16 output1_S16;
    GrayS16 output2_S16;

    // tuned to detect 3000 features in unweighted variant
    ConfigGeneralDetector configGoodFeats = new ConfigGeneralDetector(-1,10,6e4f);

    int[] histogram = new int[256];

    ConfigHoughPolar configHoughP = new ConfigHoughPolar(1,100,1,Math.PI/180.0,50,0);
    ConfigHoughFoot configHoughF = new ConfigHoughFoot(2,100,5,50,0);

    // use filter interface since it's easier to profile
    BlurFilter<GrayU8> gaussianBlur;
    InputToBinary<GrayU8> threshMean;
    ImageGradient<GrayU8, GrayS16> sobel;

    DetectDescribePoint<GrayF32, BrightFeature> surf;
    DetectDescribePoint<GrayF32, BrightFeature> sift;

    // OpenCV most likely implements just the weighted variant. That's likely because it's used for chessboard detection
    GeneralFeatureDetector<GrayU8,GrayS16> goodFeats;
    GeneralFeatureDetector<GrayU8,GrayS16> goodFeatsW;
    DetectLine<GrayU8> houghPolar;
    DetectLine<GrayU8> houghFoot;
    CannyEdge<GrayU8,GrayS16> canny;
    BinaryContourFinder contourA;

    @Setup
    public void setup() {
        BoofConcurrency.USE_CONCURRENT = concurrent;

        grayU8 = UtilImageIO.loadImage("../data/chessboard_large.jpg",GrayU8.class);
        outputU8 = grayU8.createSameShape();
        binaryU8 = grayU8.createSameShape();
        grayF32 = new GrayF32(grayU8.width,grayU8.height);
        ConvertImage.convert(grayU8,grayF32);

        output1_S16 = new GrayS16(grayU8.width,grayU8.height);
        output2_S16 = new GrayS16(grayU8.width,grayU8.height);

        // filters must be declared after USE_CONCURRENT has been set or else it won't stick
        gaussianBlur = FactoryBlurFilter.gaussian(GrayU8.class,-1,5);
        threshMean = FactoryThresholdBinary.localMean(kerLength,1.0,true,GrayU8.class);
        sobel = FactoryDerivative.sobel(GrayU8.class,GrayS16.class);

        goodFeats = FactoryDetectPoint.createShiTomasi(configGoodFeats,false,GrayS16.class);
        goodFeatsW = FactoryDetectPoint.createShiTomasi(configGoodFeats,true,GrayS16.class);
        houghPolar = FactoryDetectLineAlgs.houghPolar(configHoughP,GrayU8.class,GrayS16.class);
        houghFoot = FactoryDetectLineAlgs.houghFoot(configHoughF,GrayU8.class,GrayS16.class);

        canny = FactoryEdgeDetectors.canny(2,true, false, GrayU8.class, GrayS16.class);

        contourA = FactoryBinaryContourFinder.linearExternal();

        // tuned to detect 10,000 features with the same number of scales as the original paper
        // the fast variant is used because last I checked OpenCV's implementation had poor stability, much worse
        // than "stable" variant and a bit worse than the "fast" variant. This is more comparable.
        surf = FactoryDetectDescribe. surfFast(
                new ConfigFastHessian(14, 2, -1, 2, 9, 4, 4), null, null,GrayF32.class);

        // This tells it to use 6 octaves, since 0 to 5 is inclusive. The original paper used -1 to 5.
        // This is a bit tricky since different libraries interpret this variable differently.
        // Number of detected features was turned to be about 10,000
        sift = FactoryDetectDescribe.sift(new ConfigCompleteSift(0,5,12000));

        // TODO replace with image loaded from disk
        threshMean.process(grayU8, binaryU8);
    }

    @Benchmark
    public void gaussianBlur() {
        gaussianBlur.process(grayU8, outputU8);
    }

    @Benchmark
    public void threshMean() {
        threshMean.process(grayU8, outputU8);
    }

    @Benchmark
    public void sobel() {
        sobel.process(grayU8, output1_S16,output2_S16);
    }

    @Benchmark
    public void goodFeatures() {
        // OpenCV implements the unweighted variant
        sobel.process(grayU8, output1_S16,output2_S16);
        goodFeats.process(grayU8,output1_S16,output2_S16,null,null,null);
//        System.out.println("Shi-Tomasi detected "+goodFeats.getMaximums().size);
    }

//    @Benchmark
//    public void goodFeaturesWeighted() {
//        sobel.process(grayU8, output1_S16,output2_S16);
//        goodFeatsW.process(grayU8,output1_S16,output2_S16,null,null,null);
//    }

    // TODO tune
    @Benchmark
    public void houghPolar() {
        houghPolar.detect(grayU8);
    }

    // TODO tune
    @Benchmark
    public void houghFoot() {
        houghFoot.detect(grayU8);
    }

    // TODO tune
    @Benchmark
    public void canny() {
        canny.process(grayU8,10,100,null);
    }

    @Benchmark
    public void sift() {
        sift.detect(grayF32);
//        System.out.println("SIFT Detected = "+sift.getNumberOfFeatures());
    }

    @Benchmark
    public void surf() {
        surf.detect(grayF32);
//        System.out.println("SURF Detected = "+surf.getNumberOfFeatures());
    }

    // TODO tune
    @Benchmark
    public void contourExternal() {
        contourA.process(binaryU8);
    }

    @Benchmark
    public void histogram() {
        ImageStatistics.histogram(grayU8,0,histogram);
    }

}
