# Mask & Temperature Check System with RaspberryPi

* 프로젝트 제작 중
* 2021.10.04 ~
* 추후 완성후에 리뷰 및 참고했던 사이트 정리예정(해라시환)
* 머신러닝 모델 -> 딥러닝(FasterRCNN -> SSD Mobilenet V2)
  * 가벼운 Classification 으로 안한이유 : 정확도가 떨어져서
* 무거운 딥러닝모델을 사용할 수 있엇던 이유
  * 모션감지를 통해 과부화를 최소화 했음
  * tensorlite를 사용
  * Optimize로 양자화 하였음
