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
 * Created by Admin on 21.05.2018.
 */
public class RecognitionWithOpenCV {

    final File faceCascadeFile = new File("src\\main\\resources\\classifier\\haarcascade_frontalface_alt.xml");
    final File videoFile = new File("src\\main\\resources\\work.mp4");
    final File imagesDir = new File("src\\main\\resources\\new_faces\\");


    opencv_objdetect.CvHaarClassifierCascade classfierFace = null;
    opencv_face.FaceRecognizer faceRecognizer = createLBPHFaceRecognizer(); // класс распознования
    Map<Integer, String> ass = new TreeMap<>(); //именнованный массив ключ - значени (label - название файла)
    int count = 0;



    public static void main(String[] args) throws IOException {

        RecognitionWithOpenCV vision = new RecognitionWithOpenCV();
    }

    public RecognitionWithOpenCV() throws IOException {

        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage(); //Frame to IplImage
        IplImage img = null;

        classfierFace = cvLoadHaarClassifierCascade(faceCascadeFile.getCanonicalPath(), cvSize(0, 0)); //Load cascadclassifier
       // OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0); // Захват видео с камеры по умолчани
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFile);
        grabber.setAudioStream(0);
        grabber.start();
        grabber.setFrameNumber(300);
        Frame frame = grabber.grab(); // вывод камеры в фрейм
     //   img = converter.convert(frame);
        CanvasFrame canvasFrame = new CanvasFrame("Frame");
        canvasFrame.setCanvasSize(frame.imageWidth, frame.imageHeight);
        train();
        while (canvasFrame.isVisible() && (frame = grabber.grab()) != null) {
            img = toGray(converter.convert(frame));
        //
        //   IplImage resizeImage = getSubImageFromIpl(img, 100, 100, 200, 200); //передаем интересующие нас координаты и размеры вызваной функции
            findObject(img);
            canvasFrame.showImage(converter.convert(img));
          //  recorder.record(frame);
        }
       // recorder.stop();
    //    recorder.close();
        canvasFrame.dispose();
    }

    public void train(){
        FilenameFilter imgFilter = new FilenameFilter() { //указываем фильтр чтобы считать файлы с нужным разрешением
            @Override
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg");
            }
        };

        File[] imageFiles  = imagesDir.listFiles(imgFilter);
        MatVector images = new MatVector(imageFiles.length); // помещаем в массив

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();
        int counter = 0;

        for (File image : imageFiles){ //создаем ассоциации
            Mat baseImage = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int label = Integer.parseInt(image.getName().split("\\-")[0]);
            ass.put(label, image.getName()); //соответсвие лейбла и имейджа
            images.put(counter, baseImage); // порядковый номер и название
            labelsBuf.put(counter, label); //порядковый номер и циферка
            counter++;
            }

            faceRecognizer.train(images, labels);
            faceRecognizer.save("src\\main\\resources\\result_train_work_video.xml");
         //   faceRecognizer.;
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



    public void findObject(IplImage currentFrame){

        CvMemStorage storage = CvMemStorage.create();
        CvSeq faces = cvHaarDetectObjects(currentFrame, classfierFace, storage, 1.1, 5, CV_HAAR_MAGIC_VAL);
        int total = faces.total();
        int face_w = 64;
        int face_h = 64;
        if (total > 0) {

            System.out.println(total + " faces");

            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                IplImage face = getSubImageFromIpl(currentFrame, x, y, w, h); //берем лицо из фрейма
                face = resizeIplImage(face, face_w, face_h); //приводим к размеру

                int label = faceRecognizer.predict(cvarrToMat(face));//предсказываем, что изображено на лице
                System.out.println(String.valueOf(label));

                IplImage recFace = cvLoadImage("src\\main\\resources\\new_faces\\" + ass.get(label), CV_LOAD_IMAGE_GRAYSCALE); //загружаем предсказание по лейблу
                recFace = resizeIplImage(recFace, 50, 50); //меняем размер предсказанной картинке
                //проверка, что изображение не вылазиет за рамки:
                if ((x + recFace.width() < currentFrame.width()) && (y + recFace.height() < currentFrame.height())){
                    cvSetImageROI(currentFrame, cvRect(x, y, recFace.width(), recFace.height())); //вставлем окно внутри окна
                    cvCopy(recFace, currentFrame); //копируем в окно предсказанную картинку

                    //Надпись
                    CvFont myFont = new CvFont();
                    cvInitFont(myFont, CV_FONT_HERSHEY_SIMPLEX, 2.5, 2.5, 0.0, 3,8);
                    cvPutText(currentFrame, ass.get(label), cvPoint(10, 100), myFont, CvScalar.WHITE);
                }
             //   rectangle(cvarrToMat(currentFrame), new Rect(x,y,w,h), new Scalar (0, 255, 0, 0), 2, 0, 0);// обводим лица прямоугольничком
            }
        }

    }

    public IplImage toGray(IplImage img){
        IplImage currentFrame = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
        cvCvtColor(img, currentFrame, CV_RGB2GRAY);
        return currentFrame;
    }
}
