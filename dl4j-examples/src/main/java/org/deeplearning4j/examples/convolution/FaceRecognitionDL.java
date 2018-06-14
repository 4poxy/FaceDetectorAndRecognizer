package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;

public class FaceRecognitionDL {

   public static int height = 64;
   public static int width = 64;
   public static int channels = 3;
   public static File locationSavedModel = new File("src\\main\\resources\\models\\modelSAfaces.zip"); //Выбираем модель
   public static MultiLayerNetwork model;

    static {
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(locationSavedModel);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<Integer> labelList = Arrays.asList(0,1);

    public FaceRecognitionDL() throws IOException {
    }

    public static BufferedImage IplImageToBufferedImage(opencv_core.IplImage src) {
        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        Frame frame = grabberConverter.convert(src);
        return paintConverter.getBufferedImage(frame,1);
    }

   public static int recognizeDL (opencv_core.IplImage face) throws IOException {

       NativeImageLoader loader = new NativeImageLoader(height, width, channels);
       INDArray image = loader.asMatrix(IplImageToBufferedImage(face));
       DataNormalization scaler = new ImagePreProcessingScaler(0,1);
       scaler.transform(image);
       // Pass through to neural Net
       INDArray output = model.output(image);
       System.out.println(output.toString());
       return FaceRecognitionDL.getMaxvalue(output);
   }

    public static void main(String[] args) throws IOException {

        //Прогон тестовой фотографии

        FaceRecognitionDL.recognizeDL(cvLoadImage("src\\main\\resources\\faces3\\anya\\2011-new.jpg"));
    }

    public static int getMaxvalue(INDArray output){
        double maxValue = 0;
        int maxClass = 0;
        for (int i = 0; i < output.length(); i++) {
            if (output.getDouble(i) >= maxValue){
                maxValue = output.getDouble(i);
                maxClass = i;
            }
        }
        System.out.println(maxClass);
        return maxClass;
    }
}
