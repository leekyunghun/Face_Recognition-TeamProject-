# 종합_2 load_model
from cv2 import cv2
import numpy as np
import time

def face_classifier():
    xml_path1 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    xml_path2 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_profileface.xml"
    xml_path3 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml"
    xml_path4 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt_tree.xml"
    xml_path5 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

    face_classifier  = cv2.CascadeClassifier(xml_path1)
    face_classifier2 = cv2.CascadeClassifier(xml_path2)
    face_classifier3 = cv2.CascadeClassifier(xml_path3)
    face_classifier4 = cv2.CascadeClassifier(xml_path4)
    face_classifier5 = cv2.CascadeClassifier(xml_path5)

    return face_classifier, face_classifier2, face_classifier3, face_classifier4, face_classifier5

def OpenCV_show(frame):
    cv2.imshow('Face Cropper', frame)

def OpenCV(video, model):
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 movie.get(3)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 movie.get(4)
    print('프레임 너비: %d, 프레임 높이: %d' %(width, height))

    # fourcc = cv2.VideoWriter_fourcc('H','2','6','4') # 코덱 정의
    # out = cv2.VideoWriter(saveFilePath, fourcc, 24.0, (int(width), int(height))) # VideoWriter 객체 정의

    #Check that the file is opened
    # if video.isOpened() == False: #동영상 핸들 확인
    #     print('Can\'t open the File' + (FilePath))
    #     exit()

    # movie2 = cv2.VideoCapture(saveFilePath)
    # start = time.time()
    # endtime = int(start) + int(time2)
    # while True:
        #카메라로 부터 사진 한장 읽기 
    
    def face_detector(img, face_classifier, size = 0.5):
        # color = cv2.cvtColor(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier[0].detectMultiScale(gray,1.3,5)
        if faces is():
            faces = face_classifier[1].detectMultiScale(gray,1.3,5)
            if faces is():
                faces = face_classifier[2].detectMultiScale(gray,1.3,5)
                if faces is():
                    faces = face_classifier[3].detectMultiScale(gray,1.3,5)
                    if faces is():
                        faces = face_classifier[4].detectMultiScale(gray,1.3,5)
                        if faces is():
                            return img,[]                      
        rr = [] 
        xx = []   
        yy = []   

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            rr.append(roi)
            xx.append(x)
            yy.append(y)     

        return img,rr,xx,yy 

    ret, frame = video.read()
        
    try:
        image, face, x, y = face_detector(frame, face_classifier())
    except:
        image, face = face_detector(frame, face_classifier())
    try:
        Training_Data = []

        for i in range(len(face)): 
            face[i] = cv2.cvtColor(face[i], cv2.COLOR_BGR2GRAY)
        face = np.array(face)
        face = face.reshape(face.shape[0],200,200,1)

        #학습한 모델로 예측시도
        result = model.predict(face)
        result = np.argmax(result,axis=1)
        bss1 = []
        for i in range(len(result)):
            if result[i] == 0:
                bss1.append([cv2.putText(image, "Bae", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

            elif result[i] == 1:
                bss1.append([cv2.putText(image, "Nam", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

            elif result[i] == 2:
                bss1.append([cv2.putText(image, "Ha", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

            elif result[i] == 3:
                bss1.append([cv2.putText(image, "Kang", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

            else:
                bss1.append([cv2.putText(image, "U", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)])

        bss1[:]      
        # OpenCV_show(image)
    except:
        #얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # OpenCV_show(image)
        pass

    return image

        # # out.write(image)    
        # if cv2.waitKey(1)==13:
        #     break

        # nowtime = time.time()
        # if nowtime == endtime or nowtime >= endtime:
        #     break
    
    # video.release()
    # movie2.release()
    # out.release()
    # cv2.destroyAllWindows()

