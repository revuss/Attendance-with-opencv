from time import time
from turtle import shape
import face_recognition
import cv2
from markupsafe import re
import numpy as np
import os
import datetime 

# _________________________________________________________________________________________________________________________________________________________________________________________________________________________

video_capture = cv2.VideoCapture(0)

path = 'ImagesAttendance'


images = []
known_face_names = []
myList = os.listdir(path)


known_face_encodes=[]

for cl in myList:
    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    cv2.cvtColor(curImg,cv2.COLOR_BGR2RGB)
    images.append(curImg)
    encode = face_recognition.face_encodings(curImg)[0]
    known_face_encodes.append(encode)
    known_face_names.append(os.path.splitext(cl)[0])

# __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

print(known_face_names)
print(len(known_face_encodes))

# ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

def markAttendance(name):
      with open('Attendance.csv', 'a') as f:
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                f.writelines(f'\n{name},{date_time_string}')

# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodes, face_encoding)
            name = "Unknown"

# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
         
            face_distances = face_recognition.face_distance(known_face_encodes, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

        cv2.rectangle(frame,(0,0),(150,30),(15,76,92),-1)

        cv2.rectangle(frame,(500,0),(640,30),(15,76,92),-1)
        

        font = cv2.FONT_HERSHEY_DUPLEX


        cv2.putText(frame,' Q/esc = QUIT',(0,20),font,0.5,(115,176,192),2)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame,'  V to verify',(500,20),font,0.5,(0,0,0),2)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    cv2.imshow('Video', frame)
    
    key =cv2.waitKey(1)
    if (key == ord('q') or  key == 27):
        break
    elif (key == ord('v') or key == ord('V')):
        markAttendance(name)
        print("verified")   
    
video_capture.release()
cv2.destroyAllWindows()
