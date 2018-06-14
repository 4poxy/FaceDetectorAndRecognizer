package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.*;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.Map;
import java.util.TreeMap;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_MAGIC_VAL;
import static org.bytedeco.javacpp.opencv_objdetect.cvLoadHaarClassifierCascade;


/**
 * Created by Admin on 21.05.2018.                  D:\УЧЕБА\Курсовой проект\1\dl4j-examples\dl4j-examples\src\main\resources\classifier\
 */
public class CreateDataSet {

    final File faceCascadeFile = new File( "src\\main\\resources\\classifier\\haarcascade_frontalface_alt.xml"); //System.getProperty("user.dir")+"\\src\\main\\resources\\classifier\\haarcascade_frontalface_alt.xml");

    final File videoFile = new File("src\\main\\resources\\trainvideo.mp4");
    final String imagesDir = new String("src\\main\\resources\\S&A\\");

    opencv_objdetect.CvHaarClassifierCascade classfierFace = null;
    int count = 0;

    public static void main(String[] args) throws IOException {
        System.out.println(System.getProperty("user.dir")+ "\\src\\main\\resources\\classifier\\haarcascade_frontalface_alt.xml");
        CreateDataSet vision = new CreateDataSet();
    }

    public CreateDataSet() throws IOException {

        IplImage img = null;
        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage(); //Frame to IplImage
        classfierFace = cvLoadHaarClassifierCascade(faceCascadeFile.getCanonicalPath(), cvSize(0, 0)); //Load cascadclassifier

        //  Захват видео с камеры (камера по умолчанию - 0)
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);

        //  Захват видео из видеофайла:
        //  FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFile);

        grabber.setAudioStream(0);
        grabber.start();
        //   Для захвата из видео с определенного кадра(300):
        //  grabber.setFrameNumber(300);
        Frame frame = grabber.grab(); // вывод камеры в фрейм

        CanvasFrame canvasFrame = new CanvasFrame("Frame");
        canvasFrame.setCanvasSize(frame.imageWidth, frame.imageHeight);


        //  Для записи видео из фрейма:
        /*
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder("D:\\JAVA\\Learning\\Video\\work.mp4", frame.imageWidth, frame.imageHeight, frame.audioChannels);
        recorder.setFrameRate(25);
        recorder.setVideoCodec(13);
        recorder.setFormat("avi");
        double quality = 10;
        recorder.setVideoBitrate((int) (quality * 1024 * 1024));
        recorder.start();
        */

        while (canvasFrame.isVisible() && (frame = grabber.grab()) != null) {

            //  Вариант с цветным изображением:
            img = converter.convert(frame);

            //  Вариант с серым изображением:
            //  img = toGray(converter.convert(frame));

            //  Для захвата поределенного кусочка из видео - не используется:
            //  IplImage resizeImage = getSubImageFromIpl(img, 100, 100, 200, 200); //передаем интересующие нас координаты и размеры вызваной функции

            //  Переходим к поиску лиц:

            findObject(img);

            canvasFrame.showImage(converter.convert(img));

            //  для случая записи видео:
            //  recorder.record(frame);
        }
        // recorder.stop();
        // recorder.close();
        canvasFrame.dispose();
    }


    public void findObject(IplImage currentFrame) {

        //  Создаем временное хранилище:
        CvMemStorage storage = CvMemStorage.create();

        //  С помощью классификатора Хаара находим лицо:
        CvSeq faces = cvHaarDetectObjects(currentFrame, classfierFace, storage, 1.5, 5, CV_HAAR_MAGIC_VAL);
        int total = faces.total();
        int face_w = 64;        //1 - изображение, 2 классификатор, 3 - временное хранилище, 6 - алгоритм,  )
        int face_h = 64;        //4 - коэфициент масштабирования (1,1 - больше ложных срабатываний, замедление) (1.5 - более строго подходит к запросу и уменьшает себе работу),
        if (total > 0) {        //5 - про большом параметре(например 101) совсем срабатываний ложных не будет, зато лицо определиться ток в фас и при хорошем освещении, при маленьком знач - много  ложных срабатываний

            System.out.println(total + " faces");

            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                IplImage face = getSubImageFromIpl(currentFrame, x, y, w, h); //берем лицо из фрейма
                face = resizeIplImage(face, face_w, face_h); //приводим к размеру

                //  Сохраняем лица по указанному пути:
                cvSaveImage(imagesDir + count + "-new.jpg", face);
                count++;

                //  Обводим лицо прямоугольником:
                rectangle(cvarrToMat(currentFrame), new Rect(x, y, w, h), new Scalar(0, 255, 0, 0), 2, 0, 0);
            }
        }
    }

    public IplImage toGray(IplImage img) {

        IplImage currentFrame = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
        cvCvtColor(img, currentFrame, CV_RGB2GRAY);
        return currentFrame;
    }

    public IplImage getSubImageFromIpl(IplImage img, int x, int y, int w, int h) {

        IplImage resizeImage = IplImage.create(w, h, img.depth(), img.nChannels()); //создаем новое изображение с нужными размерами и настройками фотографии с кот. будем работать
        cvSetImageROI(img, cvRect(x, y, w, h)); //укажем с помощью метода  cvSetImageROI с какой областью фотографии будем работать
        cvCopy(img, resizeImage); // копируем сюда эту область
        cvResetImageROI(img); //вернем настройки оригнальной фотографии - вернемся к исходной области
        return resizeImage; //вернем новосозданую картинку
    }

    public IplImage resizeIplImage(IplImage img, int w, int h) {

        IplImage resizeImage = IplImage.create(w, h, img.depth(), img.nChannels());
        cvResize(img, resizeImage);
        return resizeImage;
    }
}
