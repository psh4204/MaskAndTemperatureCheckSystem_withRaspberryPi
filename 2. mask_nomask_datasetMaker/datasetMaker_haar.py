# ** 이미지 자동 분류기 **
## 마스크 쓴사람, 안쓴사람을 haar 캐스캐이딩 데이터를 통해 분류하는 프로그램

import cv2
import numpy as np
from PIL import Image, ImageFile
import os
import shutil



### (임시)_이미지 담을 디렉토리 유무 확인 후 삭제 
path = os.path.join('image')
dir_list = os.listdir(path)
num_image = len(os.listdir(path))
# if'image' in dir_list:
#     shutil.rmtree('.\image')
#     print('image 디렉토리 삭제')
# os.mkdir('image')
# print('image 디렉토리 생성')

## 디렉토리 설정
dir = os.path.join('images_k2') # <<=== 데이터 담겨있는 디렉토리(사진섞여있음)
num = len(os.listdir(dir))
print('mask image total num:', len(os.listdir(dir)))
files = os.listdir(dir)
print(dir)

## 이미지 분류
### 캐스케이드 넣기
#### multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_mcs_mouth.xml')

i = 0

for k in range(num):
    saved_dir = os.path.join('image') # 데이터 담겨있는 디렉토리(사진섞여있음)
    saved_num = len(os.listdir(saved_dir))
    count = k
    # 임계처리를 위한 밝기 한계점( 85가 가장 좋았음 )
    bw_threshold = 85

    
    # 그레이스케일화 (임계처리를 위한-> 흰색은 확실히 감지)
    img = cv2.imread(dir + '/' + files[k], cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 평활화(grayscale)
    gray = cv2.equalizeHist(gray)

    # 임계처리( 이진 흑백화 )
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('black_and_white', black_and_white)
    # 그레이 스케일로 얼굴감지
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    # 이진화된 흑백스케일로 얼굴감지
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 3)

    # 그레이, 흑백 둘다 얼굴 감지 유무
    if(len(faces) == 0 and len(faces_bw) == 0):
        ## 아무것도 감지 안됐을때
        print('undetected')
    elif(len(faces) == 0 and len(faces_bw) == 1):
        ## 얼굴이 이진화 흑백에서 감지되었을 때 (흑백에서 감지)
        # ### 컬러에서 밝기 평활화
        # src_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # ycrcb_planes = cv2.split(src_ycrcb)
        # #### 밝기 성분에 대해서만 히스토그램 평활화 수행
        # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        # dst_ycrcb = cv2.merge(ycrcb_planes)
        # img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
        i += 1
        cv2.imwrite('./image/'+str(int(num_image) + i)+'.jpg',img)
        print('SAVE :'+'./image/'+str(int(num_image) + i)+'.jpg')
    else:
        ## 흰색마스크 검출이 안된다면
        for (x, y, w, h) in faces:
            putColor = (255,255,255)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # 입 검출_그레이스케일
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # 얼굴 감지 후 입검출이 안되었을 때 -> 마스크 감지
        if(len(mouth_rects) == 0):
            # 마스크 감지
            ### 컬러에서 밝기 평활화
            # src_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # ycrcb_planes = cv2.split(src_ycrcb)
            # #### 밝기 성분에 대해서만 히스토그램 평활화 수행
            # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
            # dst_ycrcb = cv2.merge(ycrcb_planes)
            # img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
            i += 1
            cv2.imwrite('./image/'+str(int(num_image) + i)+'.jpg',img)
            print('SAVE :'+'./image/'+str(int(num_image) + i)+'.jpg')
            putColor = (255,255,255)
        else:
            # 마스크 미감지
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    # 얼굴 감지 후 입이 감지 되었을 때 -> 마스크 안씀
                    ### 컬러에서 밝기 평활화
                    # src_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    # ycrcb_planes = cv2.split(src_ycrcb)
                    # #### 밝기 성분에 대해서만 히스토그램 평활화 수행
                    # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
                    # dst_ycrcb = cv2.merge(ycrcb_planes)
                    # img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
                    i += 1
                    cv2.imwrite('./image/'+str(int(num_image)  + i)+'.jpg',img)
                    print('SAVE :'+'/image/'+ str(int(num_image)  + i)+'.jpg')
                    putColor = (0,0,255)
                    break
        # 얼굴확인되는 곳(관심영역)에 네모 그리기
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    