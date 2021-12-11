# tkinter GUI 를 활용한 마스크, 체온 체크 시스템
# -*-coding:utf-8-*-
from tkinter import messagebox
from tkinter import *
from tkinter.simpledialog import *
from threading import Thread
from PIL import Image
from PIL import ImageTk

import os
import argparse

from numpy.core.arrayprint import str_format
import cv2
import numpy as np
import sys
import time
from threading import Lock
from threading import Thread

import importlib.util

import board
import busio as io
import adafruit_mlx90614
from time import monotonic, sleep

from glob import glob
from io import BytesIO

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from pydub import effects

import requests

# ******************** 데이터 출력 클래스 *********************
# + MLX90614_적외선온도감지센서 클래스 합침
# + 사운드 컨트롤 클래스 합침
class DataController:
    # <============ 공통 함수 ============>
    def __init__(self, freq=10000):
        # 쓰레드 락
        self.lock = Lock()
        # Loop 용 데이터 감지 변수
        self.check = False
        # 전송데이터 중복 방지용 체크 변수
        self.checkDone = False
        # 쓰레드 멈춤 변수
        self.stopped = False
        # 모듈온도 받아오는 변수
        self.ambientTemp = 0
        # 체온 받아오는 변수
        self.targetTemp = 0
        # (Out) 체온 보내는 변수
        self.targetTemp_relay = 0
        # (변수) 체온이 이상하게 체크 될때 사용될 weight 값
        self.tempWeight = 1
        # 마스크 라벨링 값 받아오는 변수
        self.maskClass = 'none'
        # (Out) 마스크 라벨링 값 체크하는 변수
        self.maskClass_check = 'none'
        # (Out) 마스크 라벨링 값 보내는 변수
        self.maskClass_relay = 'none'
        # 마스크 라벨링 점수 받아오는 변수
        self.maskScore = 0.0
        # (버퍼) # 마스크 라벨링 점수 받아오는 변수
        self.maskScore_buff = 0.0
        # 마스크 이미지 받아오는 변수 ( h, w, ch )
        self.maskImage = np.full((50, 50, 3), (64, 64, 64), dtype=np.uint8)
        # (Out) 마스크 이미지 보내는 변수
        self.maskImage_relay = np.full(
            (50, 50, 3), (64, 64, 64), dtype=np.uint8)
        # 온도 센서 init
        self.i2c = io.I2C(board.SCL, board.SDA,
                          frequency=freq)  # 10k ~ 100k Hz
        self.mlx = adafruit_mlx90614.MLX90614(self.i2c)
        # 전송되었는지 확인하는 변수
        self.sendCheck = False
        # 알람 GUI 텍스쳐 값
        self.alarmText = "환영합니다."
        # 알람 GUI 텍스쳐 색상 값
        self.alarmTextColor = "white"
        # 알람 GUI 색상 값
        self.alarmColor = "black"
        
        # 지역
        self.area_name = ""
        # 회사
        self.area_company = ""
        # 시간
        self.area_time = ""

        # 센서 가중치
        self.sensor_weight = 16

    def start(self):
        self.data_t = Thread(target=self.update).start()
        return self

    def update(self):
        self.stopped = False
        buf_realTemp = 0
        buf_temp = 0
        buf_name = 'None'
        buf_company = 'None'
        buf_sensorWeight = 0
        time.sleep(1) # i2C 값을 읽으려면 시간이 필요하다. (avoid [Errno] 121)
        while True:
            time.sleep(0.1)
            if self.stopped == True:
                return
            # 현재 모듈온도
            self.checkAmbientTemp(round(self.mlx.ambient_temperature, 1))
            ### 체온 측정
            self.checkTargetTemp(round(self.mlx.object_temperature, 1))
            ### 정보 터미널에 출력
            if (self.realTargetTemp != buf_realTemp)or(self.targetTemp!=buf_temp)or(self.area_name!=buf_name)or(self.area_company!= buf_company)or(self.sensor_weight!=buf_sensorWeight):
                print('[',str(self.area_name),str(self.area_company),']\n' 
                    ,'real:',str(self.realTargetTemp), '/virtual:',str(self.targetTemp),'/weight:',str(self.sensor_weight))
            buf_realTemp = self.realTargetTemp
            buf_temp = self.targetTemp
            buf_name = self.area_name
            buf_company = self.area_company
            buf_sensorWeight = self.sensor_weight

            # --- [이미지 및 체온 체크 알고리즘] ---
            
            # [Do Check]
            # 모션이 한번이라도 감지되면
            if ((self.check) and (self.checkDone == False)):
                print('[[[[측정중입니다]]]]')
                for i in range(10):
                    # 1초간 값을 받아온 후 마지막 값을 출력한다. (for 루프)
                    ## (In) 마스크 입력
                    ### MainLoop 에서 마스크관련 값을 받아오기 때문에
                    ### 전송할 데이터만 받아오면 된다.
                    self.maskClass_check = self.maskClass
                    time.sleep(0.1)
                
                # Relay
                ## 전달할 값들 가져오기
                self.targetTemp_relay = self.targetTemp
                self.maskClass_relay = self.maskClass_check
                ## 중복 체크 방지용 변수 True( 체크했음을 표시 )
                self.checkDone = True

                # 출력후 1.5초간 보여준다.
                ### 마스크를 착용 및 , 체온이 문제 없으면 전송
                if ((self.maskClass_check == 'with_mask') and (33 < self.targetTemp < 37.5)):
                    print('[[[[통과입니다]]]]')
                    ### 이웃 보간법으로 사진 resize
                    self.maskImage_relay = cv2.resize(self.maskImage, (230,230),interpolation=cv2.INTER_NEAREST)
                    self.alarmTextColor = "white"
                    self.alarmColor = "lime green"
                    self.alarmText = str(self.targetTemp_relay)+"\n통과입니다"
                    time.sleep(0.1)      
                    
                    cv2.imwrite("./"+self.maskClass_relay + "/" + "pass_" +
                            str(self.targetTemp_relay)+".jpg", self.maskImage_relay)
                    self.playMp3_speed("./sounds/pass.mp3",1.25)
                    self.playMp3("./sounds/pass_pls.mp3")
                    # self.ttsKR_speed("통과입니다!", 1.5)
                    
                    # 기기 정보 받아오고, POST전송할 form 만들기
                    form = {
                        'area': self.area_name,
                        'company': self.area_company,
                        'temp': self.targetTemp_relay,
                        'time': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # DB URL
                    url = '해당 주소'
                    # POST 전송
                    r = requests.post(url,data=form)
                    print("STATUS :",r.text, "| ENCODE :", r.encoding)
                    time.sleep(0.1)


                #### 마스크 미착용 시 경고
                elif ((self.maskClass_check == 'without_mask') and (33 < self.targetTemp < 37.5)):
                    ### 이웃 보간법으로 사진 resize
                    print('[[[[마스크를 다시 착용]]]]')
                    self.maskImage_relay = cv2.resize(self.maskImage, (230,230),interpolation=cv2.INTER_NEAREST)
                    self.alarmTextColor = "white"
                    self.alarmColor = "crimson"
                    self.alarmText = str(self.targetTemp_relay)+"\n마스크 재착용"
                    time.sleep(0.1)

                    cv2.imwrite("./"+self.maskClass_relay + "/" + "failMask_" +
                                str(self.targetTemp_relay)+".jpg", self.maskImage_relay)
                    self.playMp3_speed("./sounds/failed.mp3",1.75)
                    self.playMp3("./sounds/wear_mask_pls.mp3")
                    # self.ttsKR_speed("마스크를 써주세요!", 1.5)

                #### 체온 문제있을 시 경고
                elif (self.targetTemp >= 37.5):
                    ### 이웃 보간법으로 사진 resize
                    print('[[[[발열 감지]]]]')
                    self.maskImage_relay = cv2.resize(self.maskImage, (230,230),interpolation=cv2.INTER_NEAREST)
                    self.alarmTextColor = "white"
                    self.alarmColor = "crimson"
                    self.alarmText = str(self.targetTemp_relay)+"\n체온 재측정"
                    time.sleep(0.1)

                    cv2.imwrite("./"+self.maskClass_relay + "/" + "failTemp_" +
                                str(self.targetTemp_relay)+".jpg", self.maskImage_relay)
                    self.playMp3_speed("./sounds/failed.mp3",1.75)
                    self.playMp3("./sounds/temp_error.mp3")
                    # self.ttsKR_speed("발열감지! 다시측정해주세요!", 1.5)
                
                #### 초기화
                for i in range(5):
                    # 0.5초간 초기화면 대기
                    self.maskImage_relay = np.full(
                        (50, 50, 3), (64, 64, 64), dtype=np.uint8)
                    self.maskClass_relay = 'none'
                    self.maskClass_check = 'none'
                    self.targetTemp_relay = 0
                    self.check = False
                    self.sendCheck = False
                    self.alarmTextColor = "white"
                    self.alarmColor = "black"
                    self.alarmText = "환영합니다."
                

            else:
                # 오류 방지용 초기화
                self.maskImage_relay = np.full(
                    (50, 50, 3), (64, 64, 64), dtype=np.uint8)
                self.maskClass_relay = 'none'
                self.maskClass_check = 'none'
                self.targetTemp_relay = 0
                self.check = False
                self.sendCheck = False
                self.alarmTextColor = "white"
                self.alarmColor = "black"
                self.alarmText = "환영합니다."


    def stop(self):
        self.stopped = True

    # <=========== 온도 함수 ===========>
    # 모듈온도 Read 함수
    def readAmbientTemp(self):
        return self.ambientTemp
    # 측정체온 Read 함수

    def readTargetTemp(self):
        return self.targetTemp

    def readRealTargetTemp(self):
        return self.realTargetTemp
    
    # 측정 온도 Return 함수 :
    def writeTempRelay(self):
        return self.targetTemp_relay

    # <=========== 객체 감지 함수 ===========>
    # 마스크 이미지 자른후 Copy 함수 :
    def catchImage(self, image, label, ymin, xmin, ymax, xmax):
        self.checkMaskImage(image[ymin:ymax, xmin:xmax].copy())
        self.checkMaskClass(label)
    
    # 마스크 이미지 Return 함수 :
    def writeMaskRelay(self):
        return self.maskImage_relay
    
    # 마스크 클래스 Return 함수 :
    def writeMaskClassRelay(self):
        return self.maskClass_relay

    
    # <======== GUI 알람 텍스트 Return 함수 ============>
    def writeAlarmText(self):
        return self.alarmText
    def writeAlarmTextColor(self):
        return self.alarmTextColor
    def writeAlarmColor(self):
        return self.alarmColor

    # <============ 사운드 관련 함수 ============>
    # 음악 재생 함수
    def playMp3(self, songPath):
        self.music = AudioSegment.from_file(songPath, format="mp3")
        play(self.music)

    # 음악 배속 재생 함수
    def playMp3_speed(self, songPath, speed):
        self.music = AudioSegment.from_file(songPath, format="mp3")
        song_speed = self.music.speedup(
            playback_speed=speed, chunk_size=150, crossfade=25)
        play(song_speed)

    # TTS 함수
    def ttsKR(self, word):
        # gTTS로 글자 받아오기
        tts = gTTS(text=word, lang="ko", tld="co.kr", slow="False")
        # 파일포인터 지정, 바이트 정보로 encoding
        self.fp = BytesIO()
        tts.write_to_fp(self.fp)
        # 시작 바이트로 이동
        self.fp.seek(0)

        # pydub, simpleAudio
        self.say = AudioSegment.from_file(self.fp, format="mp3")
        play(self.say)

        # ffcache 파일이 생성돼서 glob wild card로 전부 삭제
        self.fileList = glob("./ffcache*")
        for self.filePath in self.fileList:
            os.remove(self.filePath)

    # TTS 배속 함수
    def ttsKR_speed(self, word, speed):
        # gTTS로 글자 받아오기
        tts = gTTS(text=word, lang="ko", tld="co.kr", slow="False")
        # 파일포인터 지정, 바이트 정보로 encoding
        self.fp = BytesIO()
        tts.write_to_fp(self.fp)
        # 시작 바이트로 이동
        self.fp.seek(0)

        # pydub, simpleAudio
        self.say = AudioSegment.from_file(self.fp, format="mp3")
        # 전부 배속
        # song = self.say._spawn(self.say.raw_data, overrides={
        #     "frame_rate": int(self.say.frame_rate * 2.0)
        # })
        # 단순 프레임을 끊어서 배속(목소리변함없음)
        song_speed = self.say.speedup(
            playback_speed=speed, chunk_size=150, crossfade=25)

        play(song_speed)
        # ffcache 파일이 생성돼서 glob wild card로 전부 삭제
        self.fileList = glob("./ffcache*")
        for self.filePath in self.fileList:
            os.remove(self.filePath)
    

    # <============ 체크 함수 ============>
    # motion 으로, 체크 Loop 값 Write
    def loopCheckMotion(self, check):
        self.check = check
    
        # 주변 온도 체크 함수
    def checkAmbientTemp(self, temp):
        self.ambientTemp = temp
        
    # 체온 체크 함수
    def checkTargetTemp(self, temp):
        # 센서 오작동이 심한 관계로, 임의의 가중치를 설정
        if temp >= 34:
            self.tempWeight = temp/11.7
            self.targetTemp = round(32 + self.tempWeight, 1)
        # 추운계절일때 특정 weight값 이상 감지되게 한다.
        elif (not(4 < int(time.strftime('%m'))<10)) and (self.sensor_weight < temp < 30) :
            self.tempWeight = temp/6.1
            self.targetTemp = round(32 + self.tempWeight, 1)
        elif temp < 34:
            self.targetTemp = round(temp, 1)
            # 체온 측정이 안되면 체크Done 해제
            self.checkDone = False

        self.realTargetTemp = round(temp, 1)

    # 마스크 라벨링 값 체크 함수
    def checkMaskClass(self, label):
        self.maskClass = label

    # 마스크 라벨링 점수 체크 함수
    def checkMaskScore(self, score):
        self.maskScore = score

    # 마스크 이미지 체크 함수
    def checkMaskImage(self, image):
        self.maskImage = image
    
    # 전송체크 Return 함수 :
    def writeSendCheck(self):
        return self.sendCheck
    
    # 현재 지역 값 받아오는 함수
    def writeAreaContents(self, input_area, input_place, input_sensor_weight):
        self.lock.acquire()
        self.area_name = input_area
        self.area_company = input_place
        self.sensor_weight = input_sensor_weight
        self.lock.release()

# ******************** 비디오 스트림 클래스 ********************
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# 쓰레드와 파이프라인을 통해 영상처리의 프레임을 2.5배 상향시킬 수 있다.

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # 캠 초기화
        # VideoCapture 클래스를 이용한 세부 지정
        # 0번카메라지정
        self.stream = cv2.VideoCapture(0)
        # 인코딩
        ret = self.stream.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*'MJPG'))
        # 화면 크기
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # 첫 실행 시 첫 프레임을 읽어온다.
        # 다음 프레임을 위한 grabbed, 현재프레임을 위한 frame
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        self.cam_t = Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # 스레드가 멈출 때까지 무한 반복
        # 지속적인 업데이트
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # 다음 프레임을 위한 grabbed, 현재프레임을 위한 frame
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # 프레임 리턴
        self.frame = cv2.flip(self.frame,0)
        return self.frame

    def stop(self):
	# 카메라스레드 및 프레임 종료 변수
        self.stopped = True


# ******************** 파서 ********************
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='graph', required=False)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.8)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1024x670')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()


# ******************** 모델, 라벨맵 읽기 ********************
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


# ******************** 텐서플로우 라이브러리 Import 하기 ********************
pkg = importlib.util.find_spec('tflite_runtime')
## 일반텐서 사용시
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
## Edge TPU 사용
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
### TPU 모델 사용시 TPU 사용하기
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# 현재 경로
CWD_PATH = os.getcwd()

# 그래프 경로
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# 라벨맵 경로
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# 라벨맵 로드
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# 첫 라벨은 ??? -> 라벨맵 작성시유의하기
if labels[0] == '???':
    del(labels[0])


# ********************  텐서라이트 모델 로드  ********************
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
# 텐서 장착~!
interpreter.allocate_tensors()


# ******************** 모델 detail 변수선언  ********************
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# ******************** 디스플레이 관련 ********************
frame_rate_calc = 1  # FPS 계산 여부 확인
freq = cv2.getTickFrequency()  # 화면 틱레이트 선언

# 쓰레드 선언
# VideoStream 클래스 초기화
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
# 데이터 관리 + 온도 모듈 클래스 초기화
dataCtr = DataController(freq=10000).start()
time.sleep(1)

# [[[MAIN 코드]]]
### "모션감지"를 위한 frameRead
grabFrame = videostream.read()
grabFrame_2 = videostream.read()
motion = False   # 모션감지용 변수

# >>>> 모션감지용 임계값(모션감지 감도) <<<<
thresholdV = 25  # 25가 적정
# 모션감지 픽셀수 임계치(화면픽셀수의 0.5%_(3%가 가장 자연스러움)) : 바뀐 픽셀을 확인하여 모션캡쳐를 한다.
motion_maxCount = int(1022*670*0.005)  

# [GUI 설계]
window = Tk()  # 인스턴스 생성

window.title("One-Q PASS")  # 제목 표시줄 추가
window.geometry("1024x700")  # 지오메트리: 너비x높이 (화면크기1024x768)
window.resizable(False, True)  # x축, y축 크기 조정 비활성화
window.configure(bg='gray15')

# 프레임 _비디오
frame_video = Frame(window, bg="black", width=1024, height=670)  # 프레임 너비, 높이 설정
frame_video.place(x=0,y=31)  # 격자 행, 열 배치
# 라벨_비디오(영상출력)
label_video = Label(frame_video)
label_video.grid(row=0,sticky=(E,W,S,N))

## 프레임_얼굴이미지
frame_face = Frame(window, width=230, height=230, highlightthickness=2 ,highlightbackground='gray35')
frame_face.place(x= 794) # 224*224
frame_face["relief"] = "solid"
# GUI_라벨_위치정보알림창
label_time = Label(frame_face, text="시간")
label_time.grid(row=0,sticky=(E,W))  # 라벨 행, 열 배치
now = time.localtime()
current_time = str(now.tm_hour)+"시 "+str(now.tm_min)+"분 "+str(now.tm_sec) + "초"    # 시간 ( 00시, 00분, 00초 )
label_time.configure(text=current_time, font=(None,15),bg='gray15',fg='white')
# 프레임_상태알림창(체온 및 Pass 여부)
label_alarm = Label(frame_face,text="환영합니다.", font=(None,33),bg='black',fg='white')
label_alarm.grid(row=1,sticky=(E,W))
## 라벨_얼굴
label_face = Label(frame_face,width=224, height=224)
label_face.grid(row=2,sticky=(E,W,S,N))
label_face.configure(image='', bg='gray25')

# GUI_라벨_위치정보알림창
label_status = Label(window, text="업소 정보 : ",highlightthickness=2 ,highlightbackground='gray35')
label_status.grid(row=0)  # 라벨 행, 열 배치
current_place = "한남대학교"    # 장소 이름 ( 기본 값 )
current_local = "대전"

# 계절 별 가중치
if (4 <int(time.strftime('%m')) < 10) :
    # 봄,여름
    season_w = 30
else :
    # 가을, 겨울
    season_w = 14

# 센서 가중치
sensor_weight = 16

buf_area = 'None'
buf_place = 'None'
buf_sensor_weight = 16

# GUI 함수
def showImage(videostream, dataCtr):
    global window, current_place, now, current_time, season_w, sensor_weight
    global grabFrame, grabFrame_2, motion, thresholdV, motion_maxCount
    global interpreter, frame_rate_calc, freq, input_mean, input_std
    global min_conf_threshold
    global buf_area, buf_place, buf_sensor_weight

    
    
    # 타이머 시작( 시작 틱레이트 초기화 ) ( FPS 계산용 )
    t1 = cv2.getTickCount()

    # 화면 읽어오기
    frame1 = videostream.read()

    # 프레임 읽기 & 예측 shape을 위한 리사이징 [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # (전처리) 퀀텀화 하지못한 모델 사용시, 퀀텀화
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    ## ---------- < 모션감지용 알고리즘 > -----------
    ### GrayScale 처리
    gray_grabFrame = cv2.cvtColor(grabFrame, cv2.COLOR_RGB2GRAY)
    # gray_grabFrame_2 = cv2.cvtColor(grabFrame_2, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ### absdiff 처리
    diff_1 = cv2.absdiff(gray_frame, gray_grabFrame)
    # 이진화 처리 (thresholdV)
    ret, diff_1_t = cv2.threshold(diff_1, thresholdV, 255, cv2.THRESH_BINARY)
    ret, diff_2_t = cv2.threshold(
        gray_frame, thresholdV, 255, cv2.THRESH_BINARY)
    ### 비트연산으로 모션확인
    bw_motion_t = cv2.bitwise_and(diff_1_t, diff_2_t)
    ### 모폴로지변환 (erode)(노이즈필터)
    moph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bw_motion_b = cv2.erode(bw_motion_t, moph)

    ### motion값 변경
    motion_count = cv2.countNonZero(bw_motion_b)  # 픽셀중 0이 아닌 픽셀을 셉니다
    if motion_count > motion_maxCount:
        motion = True
    else:
        motion = False

    
    # -------- < 객체 검출 및 바운딩박스 알고리즘 + 체온 측정 > -------
    ## 1. 온도 읽어오기(실제 측정온도에서 30도 이상 감지 시 )
    realTempCheck = dataCtr.readRealTargetTemp()
    if realTempCheck > season_w:
        ## 2. 모션감지를 한다.(오작동방지+최적화용도) : 감지임계치는 낮게설정
        if motion:
            # --- 이미지를 입력 값으로 모델을 실행하여 실제 "감지 수행" ---
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # 객체탐지 결과
            boxes = interpreter.get_tensor(output_details[0]['index'])[
                0]  # 감지된 객체 바운딩 박스 좌표
            classes = interpreter.get_tensor(output_details[1]['index'])[
                0]  # 감지된 객체 클래스
            scores = interpreter.get_tensor(output_details[2]['index'])[
                0]  # 감지된 객체 정확도 점수

            # 단일 객체 인식 코드
            # 받아온 텐서 scores 중 1.0이하 중 가장 높은값의 인덱스를 가져온다. maxScoreIndex
            # 그 인덱스를 넘겨서 객체 하나만 나오게끔 한다.
            maxScoreIndex = 0
            for i in range(len(scores)):
                if((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    if (maxScoreIndex == 0):
                        maxScoreIndex = i
                    if (scores[i] >= scores[maxScoreIndex]):
                        maxScoreIndex = i

            # 바운딩박스 및 라벨링 디스플레이
            if (scores[maxScoreIndex] > min_conf_threshold):
                ymin = int(max(1, (boxes[maxScoreIndex][0] * imH)))
                xmin = int(max(1, (boxes[maxScoreIndex][1] * imW)))
                ymax = int(min(imH, (boxes[maxScoreIndex][2] * imH)))
                xmax = int(min(imW, (boxes[maxScoreIndex][3] * imW)))

                if (classes[maxScoreIndex] == 0) :
                    # 화면에 라벨 디자인 및 입력
                    # labels 에서 객체명 가져옴
                    object_name = labels[int(classes[maxScoreIndex])]
                    label = '%s: %d%%' % (object_name, int(
                        scores[maxScoreIndex]*100))  # %퍼센트 입력
                    labelSize, baseLine = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # 폰트 및 사이즈
                    # 라벨을 창 상단에 너무 가깝게 그리지 않도록 합니다.
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0],
                                label_ymin+baseLine-10), (10, 255, 0), cv2.FILLED)  # 라벨텍스트를 입력할 박스 그리기
                    cv2.putText(frame, label, (xmin, label_ymin-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 라벨텍스트 입력
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                else : 
                    # (classes[maxScoreIndex] == 'without_mask') :
                    # 화면에 라벨 디자인 및 입력
                    # labels 에서 객체명 가져옴
                    print(classes[maxScoreIndex])
                    object_name = labels[int(classes[maxScoreIndex])]
                    label = '%s: %d%%' % (object_name, int(
                        scores[maxScoreIndex]*100))  # %퍼센트 입력
                    labelSize, baseLine = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # 폰트 및 사이즈
                    # 라벨을 창 상단에 너무 가깝게 그리지 않도록 합니다.
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0],
                                label_ymin+baseLine-10), (10, 0, 255), cv2.FILLED)  # 라벨텍스트를 입력할 박스 그리기
                    cv2.putText(frame, label, (xmin, label_ymin-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 라벨텍스트 입력
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 0, 255), 2)

                

                dataCtr.catchImage(frame1.copy(), object_name,
                                ymin, xmin, ymax, xmax)

                # 루프 모션체크를 True로 하여DataController로 보낸다.
                dataCtr.loopCheckMotion(True)

            # --- 체온 측정 ---
            # os.system('clear')
            print("모듈 온도 : " + str(dataCtr.readAmbientTemp()))
            print("측정 체온 : " + str(dataCtr.readTargetTemp()))
            print("실제 측정 체온 : " + str(dataCtr.readRealTargetTemp()))

    # # ======================================

    # 현재 기기의 정보를 클래스에 보낸다.
    if(buf_area != current_local)or(buf_place != current_place)or(buf_sensor_weight != sensor_weight):
        dataCtr.writeAreaContents(input_area=current_local, input_place=current_place, input_sensor_weight=sensor_weight)
        print('>>> 기기정보 변경(',current_local,current_place,sensor_weight,')')
    buf_area = current_local
    buf_place = current_place
    buf_sensor_weight = sensor_weight


    # 화면에 FPS 입력 및 디자인
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    # FPS 계산
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # 여태 했던 작업들 화면에 그리기
    #cv2.imshow('Object detector', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame) # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img) # ImageTk 객체로 변환

    # ImageTk에 입력
    label_video.imgtk = imgtk

    # GUI 업데이트
    ## GUI 장소 업데이트
    label_video.configure(image=imgtk)
    label_status.configure(text="현재 장소 : " + current_place, font=(None,15),bg='gray15',fg='white')
    ## GUI 시간 업데이트
    now = time.localtime()
    current_time = str(now.tm_hour)+"시 "+str(now.tm_min)+"분 "+str(now.tm_sec) + "초"    # 시간 ( 00시, 00분, 00초 )
    label_time.configure(text=current_time, font=(None,15),bg='gray15',fg='white')
    ## GUI 마스크 및 온도정보 업데이트
    alarmText = dataCtr.writeAlarmText()
    alarmTextColor = dataCtr.writeAlarmTextColor()
    alarmColor = dataCtr.writeAlarmColor()
    maskFrame = dataCtr.writeMaskRelay()
    maskFrame = cv2.cvtColor(maskFrame, cv2.COLOR_BGR2RGBA)
    img_mask = Image.fromarray(maskFrame)
    imgtk_mask = ImageTk.PhotoImage(image=img_mask)
    label_face.imgtk = imgtk_mask
    label_alarm.configure(text=alarmText, font=(None,20), bg=alarmColor,fg=alarmTextColor)
    label_face.configure(image=imgtk_mask)
    
    ## "모션감지"를 위해 새로운 grab frame을 읽어온다.
    ### n프레임당 한번 grab을 캐치한다.
    grabFrame = frame1.copy()

    label_video.after(10, showImage,videostream,dataCtr)

# 영업장 지역 변경 함수
def changeLocal():
    global current_local
    ask_place = askstring("지역 정보", "현재 지역정보를 입력하세요")
    if not((ask_place == "") or (ask_place==None)):
        current_local = ask_place

# 영업장변경 함수
def changePlace():
    global current_place
    ask_place = askstring("장소 정보", "현재 장소정보를 입력하세요")
    if not((ask_place == "") or (ask_place==None)):
        current_place = ask_place

# 정보출력 함수
def showInfo():
    messagebox.showinfo("개발자 정보","[Info]\nMask & Temp Check Checker\nVersion 1.0\n::Made By Park Si Hwan::")


# 센서 가중치 변경 함수
def changeSensorWeight():
    global sensor_weight
    ask_weight = askinteger("가중치변경","센서 가중치 입력하세요(기본 16)")
    if not((ask_weight == "") or (ask_weight==None)):
        # 최대값인 30을 넘기면 안된다.
        if ask_weight > 29:
            ask_weight = 29
        sensor_weight = ask_weight


# 종료 함수
def on_closing():
    global window
    global videostream, dataCtr
    if messagebox.askokcancel("종료", "종료하시겠습니까?"):
        videostream.stop()
        dataCtr.stop()
        time.sleep(1)
        cv2.destroyAllWindows()
        window.destroy()


# 메뉴창
mainMenu = Menu(window)
window.config(menu=mainMenu)

## 메뉴창_설정
settingMenu = Menu(mainMenu)
mainMenu.add_cascade(label="설정", menu=settingMenu)
settingMenu.add_command(label="지역 설정",command=changeLocal)
settingMenu.add_command(label="장소 설정",command=changePlace)
settingMenu.add_command(label="센서가중치변경",command=changeSensorWeight)
settingMenu.add_command(label="종료", command=on_closing)
helpMenu = Menu(mainMenu)
mainMenu.add_cascade(label="도움말", menu=helpMenu)
helpMenu.add_command(label="정보",command=showInfo)

# TK GUI는 MainLoop를 이용해서 돌린다.
showImage(videostream=videostream, dataCtr=dataCtr)
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
