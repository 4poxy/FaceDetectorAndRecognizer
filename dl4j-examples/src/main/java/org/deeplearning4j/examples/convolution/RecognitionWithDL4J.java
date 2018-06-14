package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_MAGIC_VAL;
import static org.bytedeco.javacpp.opencv_objdetect.cvLoadHaarClassifierCascade;

/**
 * Created by Admin on 21.05.2018.
 */
public class RecognitionWithDL4J {

    final File faceCascadeFile = new File(System.getProperty("user.dir"), "src\\main\\resources\\classifier\\haarcascade_frontalface_alt.xml");
     final File videoFile = new File("src\\main\\resources\\testvideo.mp4"); //Ошибка с форматом тоже означает ошибку пути
    final String imagesDir = new String("src\\main\\resources\\all_facesSA\\");

    opencv_objdetect.CvHaarClassifierCascade classfierFace = null;
    Map<Integer, String> assName = new TreeMap<>(); //именнованный массив ключ - значени (label - название файла)
    Map<Integer, String> assDL = new TreeMap<>(); //именнованный массив ключ - значени (label - название файла)
    int count = 0;



    public static void main(String[] args) throws IOException {

        RecognitionWithDL4J vision = new RecognitionWithDL4J();

    }

    public RecognitionWithDL4J() throws IOException {

        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage(); //Frame to IplImage
        IplImage img = null;

        classfierFace = cvLoadHaarClassifierCascade(faceCascadeFile.getCanonicalPath(), cvSize(0, 0));
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0); // Захват видео с камеры по умолчанию
      //  FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFile); //Захват с видео

        grabber.setAudioStream(0);
        grabber.start();
     //   grabber.setFrameNumber(350);
        Frame frame = grabber.grab(); // вывод камеры в фрейм
        CanvasFrame canvasFrame = new CanvasFrame("Frame");
        canvasFrame.setCanvasSize(frame.imageWidth, frame.imageHeight);
        createAcc(); //Составляем асоциацию


        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder("D:\\JAVA\\Learning\\Video\\anna.avi", frame.imageWidth, frame.imageHeight, frame.audioChannels);
        recorder.setFrameRate(25);
        recorder.setVideoCodec(13);
        recorder.setFormat("avi");
        double quality = 10;
        recorder.setVideoBitrate((int) (quality * 1024 * 1024));
        recorder.start();

        while (canvasFrame.isVisible() && (frame = grabber.grab()) != null) {

            //img = converter.convert(frame); //-- цветное изображение -- нужно переобучать сеть!
           img = toGray(converter.convert(frame));

            findObject(img);
            canvasFrame.showImage(converter.convert(img));
            recorder.record(frame);
        }
        recorder.stop();
        recorder.close();
        canvasFrame.dispose();
    }

    public void createAcc(){
        //Делается вручную. Сопоставляем номер класса и картинку
        assDL.put(0, "0.jpg");
        assDL.put(1, "1.jpg");

        assName.put(0, "Anya");
        assName.put(1, "Sasha");

    }

    public IplImage getSubImageFromIpl(IplImage img, int x, int y, int w, int h){

        IplImage resizeImage = IplImage.create(w, h, img.depth(), img.nChannels()); //создаем новое изображение с нужными размерами и настройками фотографии с кот. будем работать
        cvSetImageROI(img, cvRect(x, y, w, h)); //укажем с помощью метода  cvSetImageROI с какой областью фотографии будем работать
        cvCopy(img, resizeImage); // копируем сюда эту область
        cvResetImageROI(img); //вернем настройки оригнальной фотографии - вернемся к исходной области
        return resizeImage; //вернем новосозданую картинку
    }

    public IplImage resizeIplImage(IplImage img, int w, int h){
        IplImage resizeImage = IplImage.create(w, h, img.depth(), img.nChannels());
        cvResize(img, resizeImage);
        return resizeImage;
    }



    public void findObject(IplImage currentFrame) throws IOException {

        CvMemStorage storage = CvMemStorage.create();
        CvSeq faces = cvHaarDetectObjects(currentFrame, classfierFace, storage, 1.5, 5, CV_HAAR_MAGIC_VAL); //1 - изображение, 2 классификатор, 3 - временное хранилище, 6 - алгоритм,  )
        int total = faces.total();//4 - коэфициент масштабирования (1,1 - больше ложных срабатываний, замедление) (1.5 - более строго подходит к запросу и уменьшает себе работу),
        int face_w = 64;
        int face_h = 64;
        if (total > 0) {                                                                    //5 - про большом парпаментре(например 101) совсем срабатываний ложных не будет, зато лицо определиться ток в фас и при хорошем освещении, при маленьком знач - много  ложных срабатываний

            System.out.println(total + " faces");

            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                IplImage face = getSubImageFromIpl(currentFrame, x, y, w, h); //берем лицо из фрейма
                face = resizeIplImage(face, face_w, face_h); //приводим к размеру

                int label = FaceRecognitionDL.recognizeDL(face);

                 System.out.println(String.valueOf(label));
                IplImage recFace = cvLoadImage(imagesDir + assDL.get(label), CV_LOAD_IMAGE_GRAYSCALE); //загружаем ассоциацию по лейблу
                recFace = resizeIplImage(recFace, 200, 200); //меняем размер предсказанной картинке
                //проверка, что изображение не вылазиет за рамки:
               if ((x + recFace.width() < currentFrame.width()) && (y + recFace.height() < currentFrame.height())){
                    cvSetImageROI(currentFrame, cvRect(x, y, recFace.width(), recFace.height())); //вставлем окно внутри окна
                 //   cvCopy(recFace, currentFrame); //копируем в окно предсказанную картинку
                   CvFont myFont = new CvFont();
                   cvInitFont(myFont, CV_FONT_HERSHEY_SIMPLEX, 1.5, 1.5, 0.0, 3,8);
                 //  cvPutText(currentFrame, String.valueOf(label), cvPoint(10, 100), myFont, CvScalar.BLUE); //Вписываем номер класса
                   cvPutText(currentFrame, assName.get(label), cvPoint(10, 100), myFont, CvScalar.BLUE); //вписываем имя

                    //Надпись
                }
                rectangle(cvarrToMat(currentFrame), new Rect(x,y,w,h), new Scalar (0, 255, 0, 0), 2, 0, 0);// обводим лица прямоугольничком
            }
        }

    }

    public IplImage toGray(IplImage img){
        IplImage currentFrame = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
        cvCvtColor(img, currentFrame, CV_RGB2GRAY);
        return currentFrame;
    }
}
