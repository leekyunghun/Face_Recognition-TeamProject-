import face_recognition
from cv2 import cv2
import numpy as np
import sys

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
# video_capture = cv2.VideoCapture("D:/Sample6.mp4")

def load_ImageData():
    # Load a sample picture and learn how to recognize it.
    suzy_image = face_recognition.load_image_file("D:/ImgDetection/KHL/data/train/0/2.jpg")
    suzy_face_encoding = face_recognition.face_encodings(suzy_image)[0]

    # Load a second sample picture and learn how to recognize it.
    juhyuk_image = face_recognition.load_image_file("D:/ImgDetection/KHL/data/train/1/2.jpg")
    juhyuk_face_encoding = face_recognition.face_encodings(juhyuk_image)[0]

    sunho_image = face_recognition.load_image_file("D:/ImgDetection/KHL/data/train/2/20.jpg")
    sunho_face_encoding = face_recognition.face_encodings(sunho_image)[0]

    hanna_image = face_recognition.load_image_file("D:/ImgDetection/KHL/data/train/3/58.jpg")
    hanna_face_encoding = face_recognition.face_encodings(hanna_image)[0]


    # Create arrays of known face encodings and their names
    known_face_encodings = [
        suzy_face_encoding,
        juhyuk_face_encoding,
        sunho_face_encoding,
        hanna_face_encoding
    ]
    known_face_names = [
        "Suzy",
        "Juhyuk",
        "Sunho",
        "Hanna"
    ]
    return suzy_face_encoding, juhyuk_face_encoding, sunho_face_encoding, hanna_face_encoding, known_face_encodings, known_face_names

def make_Emptylist():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    return face_locations, face_encodings, face_names, process_this_frame

    # while True:

def Dlib(video_capture, empty_list, data_list):
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if empty_list[3]:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(data_list[4], face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(data_list[4], face_encoding)
                print("face_distances : ", face_distances)
                best_match_index = np.argmin(face_distances)
                print("best_match_index : ", best_match_index)
                if matches[best_match_index]:
                    name = data_list[5][best_match_index]

                face_names.append(name)

        process_this_frame = not empty_list[3]


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left-10, top-10), (right+10, bottom+10), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left-10, bottom - 10), (right+10, bottom+10), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (right - 50, bottom + 6), font, 0.5, (255, 255, 255), 1)
        
        return frame

def Dlib_show(frame):
    # Display the resulting image
    cv2.imshow('Video', frame)

#         # Hit 'q' on the keyboard to quit!
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()