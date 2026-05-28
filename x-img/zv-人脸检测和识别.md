# 人脸检测和识别

## 目的和要求

1. 掌握OpenCV图像获取、读写图像文件的方法；
2. 掌握人脸检测和识别的方法和流程；
3. Python面向编程的设计思想；

## 实验设备与准备

- 计算机：CPU四核i7-6700处理器；内存8G；SATA硬盘2TB硬盘；Intel芯片主板；集成声卡、千兆网卡、显卡；20寸液晶显示器。
- 环境：Python3.14、VSCode、OpenCV4.0等

## 实验内容

通过OpenCV+Python实现对于人脸检测和人脸识别，要求掌握人脸识别的流程：人脸检测、人脸数据采集、选择人脸识别模型、模型训练、模型验证等过程。

## 实现过程

### 预备知识

熟悉和掌握OpenCV人脸检测函数和Haar级联分类器的有关知识。

### 静态图像中的人脸检测

```python
mport cv2

filename = '../images/vikings.jpg'

def detect(filename):
  face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
  img = cv2.imread(filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.09, 5,minSize = (3,3))
  for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

  cv2.namedWindow('Vikings Detected!!')
  cv2.imshow('Vikings Detected!!', img)
  cv2.imwrite('./vikings.jpg', img)
  cv2.waitKey(0)

detect(filename)
```

### 视频人脸检测

```python
import cv2

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
while (True):
    read, img = camera.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("camera", img)
    if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
        break
```

### 人脸数据采集

```python
import cv2


def generate():
  face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
  camera = cv2.VideoCapture(0)
  count = 0
  while (True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        if(count<=20):
            cv2.imwrite('../data/at/wb/%s.pgm' % str(count), f)
        print(count)
        count += 1

    cv2.imshow("camera", frame)
    if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  generate()
```

### 人脸模型训练

加载数据，从文件路径中加载图像数组，把数据加载到人脸识别的算法中，以训练人脸识别算法。把数据形成图像数据和标签对，传递到训练算法中，进行训练并保存模型。

### 人脸模型验证

对于训练模型进行验证，加载模型文件，传递图像数据并检验识别是否正确。

### 代码集成

在实验项目1代码基础上，实现人脸检测和人脸识别，把人脸检测代码和人脸识别代码整理到项目1的代码中，形成一个完整的人脸检测和识别的模块。注意：人脸识别范围两个步骤：

1. 人脸训练（保存人脸识别模型），可以理解为离线训练；
2. 人脸识别（在项目1代码中直接加载训练好的模型文件，进行识别）

## 总结和要求

- 通过本项目要掌握OpenCV人脸检测和人脸识别的方法和实现流程，理解机器学习的流程方法。
- 形成一个完整的实验报告。
