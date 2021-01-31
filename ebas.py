from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QDate
from PyQt5.QtWidgets import QDialog
from datetime import datetime, timedelta
import cv2
import face_recognition
import numpy as np
import os
import dlib
from scipy.spatial import distance


class Ui_OutputDialog(QDialog):
    def __init__(self):

        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        # current date & time
        now = QDate.currentDate()
        currentDate = now.toString('ddd dd MMM yyyy')
        currentTime = datetime.now().strftime("%I:%M %p")
        self.labelDate.setText(currentDate)
        self.labelTime.setText(currentTime)
        self.image = None
        self.lastEntry = []

    @pyqtSlot()
    def startVideo(self):
        """
        :param camera_name: link of camera or usb camera
        :return:
        """

        self.capture = cv2.VideoCapture(0)
        # if self.capture.isOpened():
        #     print('camera opened')
        # else:
        #     print('camera not opened')
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #print('dlib loaded')
        path = 'ImagesAttendance'
        if not os.path.exists(path):
            os.mkdir(path)

        # known face encoding and known face name list
        images = []
        self.class_names = []
        self.encode_list = []
        attendance_list = os.listdir(path)
        for cl in attendance_list:
            cur_img = cv2.imread(f'{path}/{cl}')
            images.append(cur_img)
            self.class_names.append(os.path.splitext(cl)[0])
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img)
            encodes_cur_frame = face_recognition.face_encodings(img, boxes)[0]
            self.encode_list.append(encodes_cur_frame)

        while True:
            ret, self.image = self.capture.read()
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.displayImage(self.image, self.encode_list, self.class_names, 1)
            cv2.waitKey(0)


    def face_rec_(self, frame, encode_list_known, class_names):
        """
        :param frame: frame from camera
        :param encode_list_known: known face encoding
        :param class_names: known face names
        :return:
        """

        def calculate_EAR(eye):
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
            ear_aspect_ratio = (A + B) / (2.0 * C)
            return ear_aspect_ratio

        def detectBlink(self,name):

            faces = self.hog_face_detector(self.gray)
            for face in faces:

                face_landmarks = self.dlib_facelandmark(self.gray, face)
                leftEye = []
                rightEye = []

                for n in range(36, 42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x, y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                for n in range(42, 48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x, y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)

                EAR = (left_ear + right_ear) / 2
                EAR = round(EAR, 2)
                if EAR < 0.26:
                    print(name + ' -> Blink detected')
                    mark_attendance(name)
            return frame



        def mark_attendance(name):
            """
            :param name: detected face known or unknown one
            :return:
            """
            # datetime.now for current date and time
            # strftime for format
            curTime = datetime.now().strftime('%H:%M:%S')
            attendancePath = 'Attendance List'
            if not os.path.exists(attendancePath):
                os.mkdir(attendancePath)
            fileName = QDate.currentDate().toString('dd-MM-yy')+'.csv'
            if not os.path.isfile(fileName):
                with open(os.path.join(fileName),'w') as k:
                    k.writelines('Name,Time')
            with open(fileName, 'r+') as f:
                flag = 0
                if len(self.lastEntry) != 0:
                    if name == self.lastEntry[0] and (datetime.strptime(curTime, '%H:%M:%S') -
                                    datetime.strptime(self.lastEntry[1], '%H:%M:%S')) < timedelta(minutes=2):
                        print('Can\'t make entry again before 2 minutes')
                        flag = -1
                if flag == 0:
                    f.writelines(f'\n{name},{curTime}')
                    print(name + ' Marked')
                    self.labelStatus.setText('Marked')
                    if len(self.lastEntry) == 0:
                        self.lastEntry.append(name)
                        self.lastEntry.append(curTime)
                    else:
                        self.lastEntry[0] = name
                        self.lastEntry[1] = curTime

        # face recognition
        faces_cur_frame = face_recognition.face_locations(frame)
        encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)
        for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
            match = face_recognition.compare_faces(encode_list_known, encodeFace, tolerance=0.50)
            face_dis = face_recognition.face_distance(encode_list_known, encodeFace)
            best_match_index = np.argmin(face_dis)
            if match[best_match_index]:
                name = class_names[best_match_index].split('-')[0]
                rollNo = class_names[best_match_index].split('-')[1]
                self.labelName.setText(name)
                self.labelRollNo.setText(rollNo)
                frame = detectBlink(self, name)

        return frame

    def displayImage(self, image, encode_list, class_names, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        image = cv2.resize(image, (640, 480))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)