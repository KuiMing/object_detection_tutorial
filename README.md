# Object Detection Tutorial

---

## Object Detector architecture


1. Input
2. Backbone: Extract essential features
3. Neck: collect and combine essential features
4. Head: Prediction

![](https://i.imgur.com/0Vy6u8o.png)
<font size=1 color='gray'>from: https://arxiv.org/abs/2004.10934</font>


---

## Prediction Concept

![](https://i.imgur.com/EdWuUJL.png)


---

## Performance of Object Detection Model

- Precision
- Recall
- IoU: Intersection over Union
- Average Precision
- mAP: mean of Average Precision


---

### Confusion matrix



![](https://i.imgur.com/7QZxs3k.png)


### Precision & Recall
$$
Precision = {TP \over (TP+FP)}, Recall = {TP \over (TP+FN)}
$$

---

## Determine Positive 

- Confidence > confidence threshold
    - Predict correct class with high enough confidence
- IoU > IoU threshold
    - Intersection Area of prediction and real is big enough



---

### IOU

![](https://i.imgur.com/bhi1bMc.png)

---

![](https://i.imgur.com/flrN1fZ.png)


---

## Average Precision


<img src=https://i.imgur.com/04dhlD1.jpg width=80%>

<font size=1 color='gray'>https://www.caranddriver.com/features/a15148150/2008-10best-cars/</font>

---

## Average Precision

- Assume
    - IOU Threshold = 0.5
    - Confidence Threshold = 0
- Predicted cars: 
    - True Positive: 7
    - False Positive: 2
- Real cars: 10


----


| B Box | Confidence | Real  | Precision | Recall |
| ----- | ---------- | ----- | --------- | ------ |
| BB1   | 0.91       | True  | 1/1       | 1/10   |
| BB2   | 0.9        | True  | 2/2       | 2/10   |
| BB3   | 0.85       | True  | 3/3       | 3/10   |
| BB8   | 0.82       | False | 3/4       | 3/10   |
| BB4   | 0.79       | True  | 4/5       | 4/10   |
| BB5   | 0.67       | True  | 5/6       | 5/10   |
| BB6   | 0.63       | True  | 6/7       | 6/10   |
| BB9   | 0.56       | False | 6/8       | 6/10   |
| BB7   | 0.55       | True  | 7/9       | 7/10   |


----


<!-- .slide: data-background-iframe="https://kuiming.github.io/object_detection_tutorial/AP.html" -->


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
