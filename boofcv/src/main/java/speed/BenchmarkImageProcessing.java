package speed;

import boofcv.abst.feature.detect.interest.ConfigGeneralDetector;
import boofcv.abst.feature.detect.line.DetectLine;
import boofcv.abst.filter.binary.InputToBinary;
import boofcv.abst.filter.blur.BlurFilter;
import boofcv.abst.filter.derivative.ImageGradient;
import boofcv.alg.feature.detect.interest.GeneralFeatureDetector;
import boofcv.concurrency.BoofConcurrency;
import boofcv.factory.feature.detect.interest.FactoryDetectPoint;
import boofcv.factory.feature.detect.line.ConfigHoughFoot;
import boofcv.factory.feature.detect.line.ConfigHoughPolar;
import boofcv.factory.feature.detect.line.FactoryDetectLineAlgs;
import boofcv.factory.filter.binary.FactoryThresholdBinary;
import boofcv.factory.filter.blur.FactoryBlurFilter;
import boofcv.factory.filter.derivative.FactoryDerivative;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.ConfigLength;
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
    ConfigLength kerLenth = ConfigLength.fixed(radius*2+1);

    GrayU8 grayU8;
    GrayU8 outputU8;

    GrayS16 output1_S16;
    GrayS16 output2_S16;

    // see python comments for these values
    ConfigGeneralDetector configGoodFeats = new ConfigGeneralDetector(1000,2,0.0000001f);


    ConfigHoughPolar configHoughP = new ConfigHoughPolar(1,100,1,Math.PI/180.0,50,0);
    ConfigHoughFoot configHoughF = new ConfigHoughFoot(2,100,5,50,0);

    // use filter interface since it's easier to profile
    BlurFilter<GrayU8> gaussianBlur = FactoryBlurFilter.gaussian(GrayU8.class,-1,5);
    InputToBinary<GrayU8> threshMean = FactoryThresholdBinary.localMean(kerLenth,1.0,true,GrayU8.class);
    ImageGradient<GrayU8, GrayS16> sobel = FactoryDerivative.sobel(GrayU8.class,GrayS16.class);

    // OpenCV most likely implements just the weighted variant. That's likely because it's used for chessboard detection
    GeneralFeatureDetector<GrayU8,GrayS16> goodFeats = FactoryDetectPoint.createShiTomasi(configGoodFeats,false,GrayS16.class);
    GeneralFeatureDetector<GrayU8,GrayS16> goodFeatsW = FactoryDetectPoint.createShiTomasi(configGoodFeats,true,GrayS16.class);
    DetectLine<GrayU8> houghPolar = FactoryDetectLineAlgs.houghPolar(configHoughP,GrayU8.class,GrayS16.class);
    DetectLine<GrayU8> houghFoot = FactoryDetectLineAlgs.houghFoot(configHoughF,GrayU8.class,GrayS16.class);

    @Setup
    public void setup() {
        BoofConcurrency.USE_CONCURRENT = concurrent;

        grayU8 = UtilImageIO.loadImage("../data/chessboard_large.jpg",GrayU8.class);
        outputU8 = grayU8.createSameShape();

        output1_S16 = new GrayS16(grayU8.width,grayU8.height);
        output2_S16 = new GrayS16(grayU8.width,grayU8.height);
    }

//    @Benchmark
//    public void gaussianBlur() {
//        gaussianBlur.process(grayU8, outputU8);
//    }
//
//    @Benchmark
//    public void threshMean() {
//        threshMean.process(grayU8, outputU8);
//    }

//    @Benchmark
//    public void sobel() {
//        sobel.process(grayU8, output1_S16,output2_S16);
//    }

//    @Benchmark
//    public void goodFeatures() {
//        // the concurrent code here seems to be barely utilized
//        sobel.process(grayU8, output1_S16,output2_S16);
//        goodFeats.process(grayU8,output1_S16,output2_S16,null,null,null);
//    }
//
//    @Benchmark
//    public void goodFeaturesWeighted() {
//        sobel.process(grayU8, output1_S16,output2_S16);
//        goodFeatsW.process(grayU8,output1_S16,output2_S16,null,null,null);
//    }

    @Benchmark
    public void houghPolar() {
        houghPolar.detect(grayU8);
    }

    @Benchmark
    public void houghFoot() {
        houghFoot.detect(grayU8);
    }

}
