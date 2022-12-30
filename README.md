---
slideOptions:
  transition: slide
  theme: black
  spotlight:
    enabled: true
---

# Object Detection Tutorial

---

<!-- .slide: data-background="gray" -->
## https://reurl.cc/KXox0M

<img src=https://i.imgur.com/J9GAA9W.png width=60%>


---

## Speaker

- 陳奎銘 Kui-Ming Chen (Benjamin)
- Graduated from Institute of Biomedical Informatics
- Data Scientist
- Microsoft MVP in AI field
- Consultant
- Lecturer
- e-mail: benjamin0901@gmail.com

Note:
 My Name is Kui-Ming Chen, and you can call me Benjamin. I'm graduated from Institute of Biomedical Informatics, so I have some knowledge about the biology. And I'm a Data scientist right now. And I got Microsoft MVP in AI field this year. I'm also a consultant about data science. I have some teaching experience. I taught students about AI or data science. Although I have to report in English every Tuesday meeting, I never teach in English. So, If my grammar is wrong, or if you don't understand what I'm saying, I will explain in Chinese. But Don't worry, All my slides are in English. If it really doesn’t work, I can speak Chinese and ask Professor Chen to translate for me.

---

## Outline

- Object Detection Introduction
- Labeling Tool
- Train Object Detection Model with Google Colab

---

# Object Detection Introduction

---

## Object Detection

- Detect the location of objects in images
- Classify or recognize the objects in images

Note:
 Today, I want to introduce Object Detection. Object detection is a technique of computer vision and deep learning. But I'm not going to talk about deep learning. I just want to let you know the concept of object detection and how to use train and use the object detection model. 

---

## YOLO

- You only look once (YOLO)
- state-of-the-art, real-time object detection system
- We will use YOLO v4 (2020)
- The latest version is v7


Note:
 YOLO is one of object detection method. It means you only look ones. it's the state-of the art, real-time object detection system. Althogh The latest version is v7, We will use YOLO v4 today. Because I think YOLO v4 is enough, and it's easy to prepare in 2 weeks.

---


## Object Detector architecture


1. Input
2. Backbone: Extract essential features
3. Neck: collect and combine essential features
4. Head: Prediction

![](https://i.imgur.com/0Vy6u8o.png)
<font size=1 color='gray'>from: https://arxiv.org/abs/2004.10934</font>

Note:
 It's architecture of Object detection model. There are so many layers, and it consists of four parts. Input for images, and Extract essential features of images in backbone part, and collect and combine essential features in Neck part. And head part is for the prediction. Predict which part is an object and what the object is.

---

## Prediction Concept

![](https://i.imgur.com/EdWuUJL.png)


Note:
 There are two object detection type. One is dense prediction, the model predict every part of an image, and according the prediction result to gather the parts together, like this. And the other one is find the location fisrt and predict what the object is.

----


![](https://i.imgur.com/Wzf1hx4.png)



---

## Workflow for object detectior model

1. Collect Images
2. Label
3. Pre-process
4. Train
5. Evaluate
6. Deploy or inference

Note:
 First of all you need to collect images. And then label the images, tell computer where and what the objects are. And we need to do some data pre-preocess. And then spend several hours or days to train the model. After training, we have to eveluate if the model is good enough. And then use your model, no mater deploy the model as a service, or just use it to detect objects.

---

## Labeling Tool- LabelImg

https://github.com/heartexlabs/labelImg


![](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)

Note:
 I guess Profesor Chen already show you this Tool, LabelImg. You need to use this to label images and create annotation files. After you install this, you can use user interface to put your images into this software, and frame the objects in the images, and give a name for each object. And then output the xml file which has all the labeling information of each image files.


---

## Pre-process

- extract the information of labeling
- Image Augmentation
- Split data into (training, validation) and test set

Note:
 If you already have images and xml files, we can start to pre-process the data. We have to extract the information from the xml files. And sometimes, we also have to do the image augmentation by ourselves, but not this time. I'll talk about this later. And the final thing we should do is to split the data into training, validation and test set. My demo code can split the data into training and validation set. So, please prepare your test data set on your own. You can prepare 10% of the total images and labeling files as your test data.


----

## Image Augmentation

A process of creating new training examples from the existing ones
![](https://i.imgur.com/sQ5crnN.png)


Note: 
 YOLO provides some augmentation method for training. And to the augmentation automatically during training. 
 your training dataset include images with objects at different: scales, rotations, lightings, from different sides, on different backgrounds - you should preferably have 2000 different images for each class or more, and you should train 2000 iterations or more for each classes.
 
In dealing with photometric distortion, YOLO adjust the brightness, contrast, hue, saturation, and noise of an image. For geometric distortion, YOLO add random scaling, cropping, flipping, and rotating. And YOLOv4 also provides some augmentation technologies, like Mosaic, Cutmix, Mixup.


---

## Performance of Object Detection Model

- Precision
- Recall
- IoU: Intersection over Union
- Average Precision
- mAP: mean of Average Precision
- Average Loss

Note:
 From now on, let's talk a little about the concept of mathematics. Because we need some number to determine whether the model is good enough. 
 

---

### Confusion matrix



![](https://i.imgur.com/7QZxs3k.png)


### Precision & Recall
$$
Precision = {TP \over (TP+FP)}, Recall = {TP \over (TP+FN)}
$$

Note:
 The first one is confusion matrix. It is a special table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the table). If the real situation is positive, and the model predict it as positive, then it true positive. It the real situation is positive, and the model predict it as negative, it's false negative. vice versa, we have false positive and true negative in the table. So, we can fill this table with numbers which means how many cases are True Positive or other situation. 
 And we can use this table to calculate precision and recall.
 Precision is how precise when the model predict something as positive. And recall is What the percentage of real positive were predicted

----

### Precision & Recall
$$
Precision = {TP \over (TP+FP)} = {TP \over Predicted Positive}
$$


$$
Recall = {TP \over (TP+FN)} = {TP \over Real Positive}
$$

---

## Determine Positive 

- Confidence > confidence threshold
    - Predict correct class with high enough confidence
- Intersection over Union (IoU) > IoU threshold
    - Intersection Area of prediction and real is big enough

Note:
 And I'm going to tell you how to determine positive with object detection model. Object detection model will calculate the confidence value that means how much confidence the model predict what a object is. And Intersection over Union is a term used to describe the extent of overlap of the labeling you give and the bounding box the model predict. We want the Intersection Area of prediction and real is big enough

---

### IOU

![](https://i.imgur.com/bhi1bMc.png)

Note:
 If the Intersection area is totally the same with Union Area, that means your model is perfect. Your model can detect the objects like a human.

---

![](https://i.imgur.com/flrN1fZ.png)

Note:
 For example, the there is a dog in this image. and the model predict it as a dog with 0.9 confidence or 90% confidence. and the IOU seems quiet big, maybe above 70%.

---

## Average Precision


<img src=https://i.imgur.com/04dhlD1.jpg width=80%>

<font size=1 color='gray'>https://www.caranddriver.com/features/a15148150/2008-10best-cars/</font>

Note:
 Now we have concept of precision and recall and IoU, we can start to talk about Average Precision. There are 10 cars in this images, and the green boxes are from the object detection model. There are 9 boxes, and two boxes are wrong. And three cars are missed.

---

## Average Precision

- Assume
    - IoU Threshold = 0.5
    - Confidence Threshold = 0.5
- Predicted cars: 
    - True Positive: 7
    - False Positive: 2
- Real cars: 10

Note:
 Give some assumption. Assume the IoU threshold is 0.5, and the confidence threshold is also 0.5.

----


| B Box | Confidence | Real | Precision | Recall | TP  | FP  |
| ----- | ---------- | ---- | --------- | ------ | --- | --- |
| BB1   | 0.91       | Car  | 1/1       | 1/10   | 1   | 0   |
| BB2   | 0.9        | Car  | 2/2       | 2/10   | 2   | 0   |
| BB3   | 0.85       | Car  | 3/3       | 3/10   | 3   | 0   |
| BB8   | 0.82       | No   | 3/4       | 3/10   | 3   | 1   |
| BB4   | 0.79       | Car  | 4/5       | 4/10   | 4   | 1   |
| BB5   | 0.67       | Car  | 5/6       | 5/10   | 5   | 1   |
| BB6   | 0.63       | Car  | 6/7       | 6/10   | 6   | 1   |
| BB9   | 0.56       | No   | 6/8       | 6/10   | 6   | 2   |
| BB7   | 0.55       | Car  | 7/9       | 7/10   | 7   | 2   |

Note:
 And then we can calculate precision and recall. We sort the data with confidence, and every time we calculate the precision and recall of first N rows. And you can see the column Real, that means the bounding box in the images is car or not. For example, precision at the first row, there is only one object predicted as a car, and it's a real car. The precision is 1 over 1. But actually there are 10 cars, so the recall is 1 over 10. And the second row there are two real cars are detected, so the precision is two over two. And the recall is 2 over 10. The forth row there are four bounding box but only real cars are detected, so the presicion is 3 over 4, and the Recall is 3 over 10. And follow the rules we fill up the table.

----


<!-- .slide: data-background-iframe="https://kuiming.github.io/object_detection_tutorial/AP.html" -->

Note: And then we can plot the precision and recall like this. the x axis is recall and the y axis is presicion. But you can see the zigzig or the fluctuation here. But actually, we will remove the zigzag and interpolate the precision as the red one. The interpolated Precision is the maximum Precision corresponding to the Recall value greater than the current Recall value. And we calculte the area under the curve as Average Precision.

----

<!-- .slide: data-background-iframe="https://kuiming.github.io/object_detection_tutorial/AP.html" -->

```
Average Precision = 
(0.3 - 0) * 1 + 
(0.6 - 0.3) * 6/7 + 
(0.7 - 0.6) * 7/9 = 0.635
```



----

## Mean of Average Precision

- $AP_i$: AP of each class
- $n$: number of classes
$$
{1 \over n}\sum_{i=1}^{n} AP_i
$$

Note:
 And we can calculate Average Precision of each class. And then we can get mean of average precision.

---

![](https://i.imgur.com/193HYKx.png)

Note:
 If you feel pain about all the mathmatical things, you can just remeber two thing. One is we want the mAP of the model to be as high as possible. This figure is about the result of an object detection. It provides so many things, but you can find the Average Precision of each class. And mAP is 98.81 %.

---

## Average Loss
```

3002: 0.521667, 0.62731 avg loss, 0.000000 rate, 9.115691 seconds, 3264 images, 8.950291 hours left
```
- 3002 - iteration number (number of batch)
- 0.62731 avg loss - average loss (error) - the lower, the better
- The final average loss can be 0.05 ~ 3.0
    - 0.05: for a small model and easy dataset
    - 3.0: for a big model and a difficult dataset

Note:
 The other thing you should remmenber is we can find the average loss during the model training. We will want the loss to be as low as posible. But actually, it's not easy to be zero. So, if your model is small with easy dataset, the average loss around 0.05 is good enough. If your model is really big with difficult dataset, and the loss below 3 is good enough. And if the loss is lower than our threshold, we can stop training. 

----

## Average Loss


<img src=https://i.imgur.com/hKkFBj3.png width=52%>

<font size=1>https://www.researchgate.net/figure/A-chart-of-the-loss-function-in-the-cell-nucleus-images-trained-by-the-YOLO-algorithm_fig4_343521390</font>

Note:
 After we start to train, we can get the chart like this. And it will renew automatically with the progress of the training. But if you use google colab to train the model, the chart will not be so prefect, it will only plot the point of some iterations, because Google will interrupt your training from time to time. And you may restart the training from the last iteration. For example, you will restart from the 1000th iteration, and the chart will redraw from the 1000th iteration.

---

# Labeling Tool

## https://github.com/heartexlabs/labelImg

----

# Labeling Tool

- Install [Anaconda](https://www.anaconda.com/download/#download)
- Open the **`Anaconda Prompt`**
- Execute the following command in **`Anaconda Prompt`**
```bash!
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```

---

# Train Object Detection Model with Google Colab

## https://reurl.cc/4XMyQL

