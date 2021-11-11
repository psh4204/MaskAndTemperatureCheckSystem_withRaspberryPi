# ** 이미지 자동 분류기 **
## 마스크 쓴사람, 안쓴사람을 RCNN 을 통해 분류하는 프로그램

from six.moves import range
import six
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
from utils import visualization_utils as vis_util
from utils import label_map_util
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image, ImageFile
import os
import shutil
from xml.etree.ElementTree import Element, SubElement, ElementTree

import collections
import matplotlib
matplotlib.use('Agg')  # pylint: disable=multiple-statements


## 사용할 이미지 디렉토리 설정---------------------
dir = os.path.join('images_before')  # <<=== 데이터 담겨있는 디렉토리(사진섞여있음)
num = len(os.listdir(dir))
print('mask image total num:', len(os.listdir(dir)))
files = os.listdir(dir)
print(dir)

## Tensorflow 라벨링 관련-------------------------
### 이 코드는 object_detection 폴더에 저장되어 있기 때문에 필요합니다.
sys.path.append("..")
### Import utilites
### 현재 디렉토리 확인
CWD_PATH = os.getcwd()
### 사용 중인 개체 감지 모듈이 포함된 디렉토리 이름
MODEL_NAME = 'inference_graph'
### 추론그래프(.pb)위치_객체감지용
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
### 라벨맵 관련
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
### 객체 종류
NUM_CLASSES = 2
### 라벨맵 로딩
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

### 메모리에 텐서플로 모델 로드
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
### 객체 감지 분류기에 대한 입력 및 출력 텐서(즉, 데이터)를 정의합니다.
### 입력 텐서는 이미지입니다.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

## 출력 텐서는 감지 box, score 및 class입니다.
### 각 상자는 특정 개체가 감지된 이미지의 일부를 나타냅니다.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
### 점수는 클래스 레이블과 함께 결과 이미지에 표시됩니다.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

### 감지된 개체 수
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

## 라벨링 함수
# < 마스크 라벨 이름 차례로 입력 >
STANDARD_LABEL = [
    'with_mask', 'without_mask', 'none'
]


def autolabel_rcnn(
        image,
        boxes,
        classes,
        scores,
        max_boxes=10,
        min_score_thresh=.5):
    # 모든 상자 위치에 대한 표시 문자열(및 클래스)을 만들고 모든 상자를 그룹화
    box_and_class = collections.defaultdict(str)
    max_boxes = boxes.shape[0]
    for i in range(boxes.shape[0]):
    # 점수들 확인
        if max_boxes == len(box_and_class):
            break
        # scores 처리_ 최소점수 비교 후 box 에 넘파이->(튜플(리스트))
        if scores is None or scores[i] > min_score_thresh:
        # 박스 내에 xy값을 처리하기 위한 리스트화 및 리스트들 튜플화
            box = tuple(boxes[i].tolist())
        # LOOK : scores 맵처리
        ##
        box_and_class[box] = STANDARD_LABEL[
            classes[i] % len(STANDARD_LABEL)]

    # 바운더리 박스 및 라벨 추출
    for box, labels in box_and_class.items():
      ## LOOK : 박스값 받아오기 ( alpha는 투명도 )
        ymin, xmin, ymax, xmax = box  # 박스 좌표
        label = labels  # 라벨

        ### XML 라벨 입력
        point1 = (xmin, ymin)
        point2 = (xmax, ymax)
        #객체 감지 파트
        obj = SubElement(root, 'object')
        # 클래스 = 객체 = 라벨 이름
        SubElement(obj, 'name').text = label
        # 기본값
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bbox = SubElement(obj, 'bndbox')
        # xy축
        SubElement(bbox, 'xmin').text = str(point1[0])
        SubElement(bbox, 'ymin').text = str(point1[1])
        SubElement(bbox, 'xmax').text = str(point2[0])
        SubElement(bbox, 'ymax').text = str(point2[1])
        #===




# ---------------  MAIN  ------------------
for k in range(num): 
    ### 저장될 이미지, 파일 디렉토리 설정
    path = os.path.join('images')
    dir_list = os.listdir(path)
    num_image = len(os.listdir(path))
    
    
    ## [Tensorflow]
    ### 이미지 이름
    IMAGE_NAME = str(num_image)
    ### 이미지 주소
    PATH_TO_IMAGE = os.path.join(dir, IMAGE_NAME)
    ### 이미지 입력
    image = cv2.imread(PATH_TO_IMAGE)
    height,width,channel = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    ### 텐서플로우로 이미지를 입력모델을 실행하여 실제 감지 수행
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    # [XML]
    # XML_ 파일 이름은 저장될  파일 이름으로
    filename = str(num_image)
    #  라벨 시작
    root = Element('annotation')
    # 이미지랑 같이 있을 폴더이름(test, train -> images)
    SubElement(root, 'folder').text = 'images'
    # 이미지 이름과 포맷
    SubElement(root, 'filename').text = filename + '.jpg'
    # 이미지의 주소와 포맷
    SubElement(root, 'path').text = './images/' +  filename + '.jpg'
    # 기본입력되는 source
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    # 사이즈 입력
    # 너비, 포인트는 opencv에 맞게 하기
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    # Color Scale =3 , Gray Scale = 1
    SubElement(size, 'depth').text = str(channel)
    # 분할 안되어서 0
    SubElement(root, 'segmented').text = '0'

    ### 감지하기 ( min_score_ thresh = 0.60 -> 60% )
    autolabel_rcnn(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=0.60)

    
    cv2.imwrite('./images/'+str(int(num_image))+'.jpg',image)
    print('SAVE :'+'./images/'+str(int(num_image))+'.jpg')
    tree = ElementTree(root)
    tree.write('./annotations/' + filename +'.xml')
