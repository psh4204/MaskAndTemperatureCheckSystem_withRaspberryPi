import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

import board
import busio as io
import adafruit_mlx90614
from time import sleep


# ******************** 데이터 출력 클래스 *********************
# + MLX90614_적외선온도감지센서 클래스 합침
tempValue = 1 # 적외선 센서 가중치(잘안되면 만져야하는 부분)
class DataController:
    global tempValue
    def __init__(self,freq=10000) :
        # TODO : 변수 조정
        self.stopped = False
        # 주변온도 받아오는 변수
        self.ambientTemp = 0
        # 체온 받아오는 변수
        self.targetTemp = 0 
        # 마스크 라벨링 값 받아오는 변수
        self.maskClass = 0
        # 마스크 라벨링 점수 받아오는 변수
        self.maskScore = 0.0
        # 마스크 이미지 받아오는 변수 ( h, w, ch )
        self.maskImage = np.full((50,50,3), (255,255,255), dtype=np.uint8)
        # 온도 센서 init
        self.i2c = io.I2C(board.SCL, board.SDA, frequency=freq) # 10k ~ 100k Hz
        self.mlx = adafruit_mlx90614.MLX90614(self.i2c)

    def start(self) : 
        self.data_t = Thread(target=self.update).start()
        return self
    def update(self) :
        self.stopped = False
        while True :
            if self.stopped == True :
                return
            # 현재 기온
            self.ambientTemp = round(self.mlx.ambient_temperature,1)
            # 측정 온도 ( 추천 측정 거리 : 1cm )
            self.targetTemp = round(self.mlx.object_temperature * tempValue,1)
            time.sleep(0.05)
    

    def stop(self) :
        self.stopped = True

    # 주변온도 체크 함수
    def readAmbientTemp(self):
        return self.ambientTemp
    def readTargetTemp(self):
        return self.targetTemp
    # 체온 체크 함수
    def checkAmbientTemp(self, temp):
        self.ambientTemp = temp
    # 주변온도 체크 함수
    def checkTargetTemp(self, temp):
        self.targetTemp = temp

    # 마스크 라벨링 값 체크 함수
    def checkMaskClass(self, label) :
        self.maskClass = label
    
    # 마스크 라벨링 점수 체크 함수
    def checkMaskScore(self, score) :
        self.maskScore = score
    
    # 마스크 이미지 체크 함수
    def checkMaskImage(self, image) :
        self.maskImage = image
    
    # 마스크 이미지 자른후 Copy 함수 :
    def catchImage(self, image, label ,ymin, xmin, ymax, xmax):
        self.checkMaskImage(image[ ymin:ymax , xmin:xmax ].copy())
        self.checkMaskClass(label)
        # test View
        cv2.imshow(self.maskClass ,self.maskImage)
        cv2.waitKey(1)





# ******************** 비디오 스트림 클래스 ********************
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# 쓰레드와 파이프라인을 통해 영상처리의 프레임을 2.5배 상향시킬 수 있다.
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # 캠 초기화
        # VideoCapture 클래스를 이용한 세부 지정
        # 0번카메라지정
        self.stream = cv2.VideoCapture(0)
        # 인코딩
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 화면 크기
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # 첫 실행 시 첫 프레임을 읽어온다.
        # 다음 프레임을 위한 grabbed, 현재프레임을 위한 frame
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        self.cam_t = Thread(target=self.update,args=()).start()
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
        return self.frame

    def stop(self):
	# 카메라스레드 및 프레임 종료 변수
        self.stopped = True



# ******************** 파서 ********************
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.8)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='480x800')
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
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# 라벨맵 경로
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

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
frame_rate_calc = 1 # FPS 계산 여부 확인
freq = cv2.getTickFrequency() # 화면 틱레이트 선언

# 쓰레드 선언
# VideoStream 클래스 초기화
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
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
motion_maxCount = int(480*800*0.03) # 모션감지 픽셀수 임계치(화면픽셀수의 3%)

# 프레임_한개_처리코드 [camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)]:
while True:
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
    # ### 블러_양방향 필터링(bilaterFilter)
    # bf_grabFrame = cv2.bilateralFilter(grabFrame,-1, 15, 15)
    # bf_frame = cv2.bilateralFilter(frame,-1, 15, 15)
    ### GrayScale 처리
    gray_grabFrame = cv2.cvtColor(grabFrame, cv2.COLOR_RGB2GRAY)
    # gray_grabFrame_2 = cv2.cvtColor(grabFrame_2, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ### absdiff 처리
    diff_1 = cv2.absdiff(gray_frame, gray_grabFrame)
    # diff_2 = cv2.absdiff(gray_grabFrame, gray_grabFrame_2)
    # 이진화 처리 (thresholdV)
    ret, diff_1_t = cv2.threshold(diff_1, thresholdV, 255, cv2.THRESH_BINARY)
    ret, diff_2_t = cv2.threshold(gray_frame, thresholdV, 255, cv2.THRESH_BINARY)
    # ret, diff_1_t = cv2.threshold(diff_1, thresholdV, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret, diff_2_t = cv2.threshold(gray_frame, thresholdV, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # diff_1_t = cv2.adaptiveThreshold(diff_1, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
    # diff_2_t = cv2.adaptiveThreshold(gray_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
    ### 비트연산으로 모션확인
    bw_motion_t = cv2.bitwise_and(diff_1_t, diff_2_t)
    ### 모폴로지변환 (erode)(노이즈필터)
    moph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bw_motion_b = cv2.erode(bw_motion_t, moph)
    # bw_motion = cv2.morphologyEx(bw_motion_b, cv2.MORPH_OPEN, moph)

    ### motion값 변경
    motion_count = cv2.countNonZero(bw_motion_b) # 픽셀중 0이 아닌 픽셀을 셉니다
    if motion_count > motion_maxCount:
        motion = True
        # print(str(motion))
    else :
        motion = False
        # print(str(motion))
    # bw_motion = cv2.cvtColor(bw_motion_b, cv2.COLOR_GRAY2BGR)

    # -------- < 객체 검출 및 바운딩박스 알고리즘 + 체온 측정 > -------
    if motion :
        # --- 이미지를 입력 값으로 모델을 실행하여 실제 "감지 수행" ---
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # 객체탐지 결과
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # 감지된 객체 바운딩 박스 좌표
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # 감지된 객체 클래스
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # 감지된 객체 정확도 점수
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # 객체 수 

        # # 최소 임계값(minimum treashhold)이상으로 나온 객체의 바운딩 박스 표시 루프문 
        # # 다중 객체인식 코드 
        # for i in range(len(scores)):
        #     if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

        #         # 바운딩박스 좌표 가져오고 그리기
        #         ymin = int(max(1,(boxes[i][0] * imH)))
        #         xmin = int(max(1,(boxes[i][1] * imW)))
        #         ymax = int(min(imH,(boxes[i][2] * imH)))
        #         xmax = int(min(imW,(boxes[i][3] * imW)))
                
        #         cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

        #         # 화면에 라벨 디자인 및 입력
        #         object_name = labels[int(classes[i])] # lavels 에서 객체명 가져옴
        #         label = '%s: %d%%' % (object_name, int(scores[i]*100)) # %퍼센트 입력
        #         labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # 폰트 및 사이즈
        #         label_ymin = max(ymin, labelSize[1] + 10) # 라벨을 창 상단에 너무 가깝게 그리지 않도록 합니다.
        #         cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # 라벨텍스트를 입력할 박스 그리기
        #         cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # 라벨텍스트 입력


        # 단일 객체 인식 코드
        # 받아온 텐서 scores 중 1.0이하 중 가장 높은값의 인덱스를 가져온다. maxScoreIndex
        # 그 인덱스를 넘겨서 객체 하나만 나오게끔 한다.
        maxScoreIndex = 0
        for i in range(len(scores)):
            if((scores[i] > min_conf_threshold ) and (scores[i] <= 1.0)):
                if (maxScoreIndex == 0):
                    maxScoreIndex = i
                if (scores[i] >= scores[maxScoreIndex]) :
                    maxScoreIndex = i
                
        # 바운딩박스 및 라벨링 디스플레이
        if (scores[maxScoreIndex] > min_conf_threshold) :
            ymin = int(max(1,(boxes[maxScoreIndex][0] * imH)))
            xmin = int(max(1,(boxes[maxScoreIndex][1] * imW)))
            ymax = int(min(imH,(boxes[maxScoreIndex][2] * imH)))
            xmax = int(min(imW,(boxes[maxScoreIndex][3] * imW)))
        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # 화면에 라벨 디자인 및 입력
            object_name = labels[int(classes[maxScoreIndex])] # lavels 에서 객체명 가져옴
            label = '%s: %d%%' % (object_name, int(scores[maxScoreIndex]*100)) # %퍼센트 입력
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # 폰트 및 사이즈
            label_ymin = max(ymin, labelSize[1] + 10) # 라벨을 창 상단에 너무 가깝게 그리지 않도록 합니다.
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # 라벨텍스트를 입력할 박스 그리기
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # 라벨텍스트 입력

            dataCtr.catchImage(frame1.copy(), object_name, ymin, xmin, ymax, xmax)

        # --- 체온 측정 ---
        os.system('clear')
        print("현재 기온 : " + str(dataCtr.readAmbientTemp()))
        print("측정 체온 : " + str(dataCtr.readTargetTemp()))

        

    # # ======================================

    # 화면에 FPS 입력 및 디자인
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # 여태 했던 작업들 화면에 그리기
    # merged = np.hstack((frame, bw_motion))
    # cv2.imshow('Object detector', merged)
    cv2.imshow('Object detector', frame)
    # cv2.imshow('Motion detector', bw_motion)

    ## "모션감지"를 위해 새로운 grab frame을 읽어온다.
    ### n프레임당 한번 grab을 캐치한다.
    grabFrame = frame1.copy()
    # grabFrame_2 = frame1.copy()

    # FPS 계산
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # 'q'입력시 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 메모리 해제 및 종료
cv2.destroyAllWindows()
videostream.stop()
dataCtr.stop()