import CVlib_test
import Dlib_test
import OpenCV_test
from tensorflow.keras.models import load_model
from cv2 import cv2

cvlib_model_path = 'D:/ImgDetection/KHL/Checkpoint/cp-inc-rmsprop-58-0.599426.hdf5'
cvlib_model = load_model(cvlib_model_path)

OpenCV_model_path = 'D:/ImgDetection/KHL/Checkpoint/CV05_2_3_MCP_1214-23-3.8027.hdf5'
OpenCV_model = load_model(OpenCV_model_path)

video = cv2.VideoCapture("D:/Sample6.mp4")

dlib_data = Dlib_test.load_ImageData()
dlib_emptyList = Dlib_test.make_Emptylist()

cvlib_character = CVlib_test.character()

face_classifier = OpenCV_test.face_classifier()

while video.isOpened():
    cvlib_frame = CVlib_test.CVlib(video, cvlib_model, cvlib_character)
    dlib_frame = Dlib_test.Dlib(video, dlib_emptyList, dlib_data)
    opencv_frame = OpenCV_test.OpenCV(video, OpenCV_model)

    cvlib_frame = cv2.resize(cvlib_frame, (720, 480))
    dlib_frame = cv2.resize(dlib_frame, (720, 480))
    opencv_frame = cv2.resize(opencv_frame, (720, 480))

    CVlib_test.cvlib_show(cvlib_frame)
    Dlib_test.Dlib_show(dlib_frame)
    OpenCV_test.OpenCV_show(opencv_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()