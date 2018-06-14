# CS420 Machine Learning Final Homework
![Language](https://img.shields.io/badge/Language-Python3-blue.svg)
## Introduction

## Preprocessing

![image](images/preprocessing.png)

- Centered(find the center of white pixels and shift it to the center)
- Crop image to 28x28 (784 << 2025) (will drop many useless blocks)

## Data Set
you can download the dataset by click the data set name  

|Name                                       | Train Size| Test Size | Image Size| Diff|
|:--:                                           |:--:      |:--:            |:--:  |:--:|
|[origin](http://cmach.sjtu.edu.cn/course/cs420/projects/mnist.zip)| 60000| 10000| 45| original data|
|[crop]()|60000|10000|28| crop from origin|
|[crop2]()|60000|10000|14|2\*2 upsampling from crop|
|[low_conf]()|15000|10000|28|images with lowest 15000 conidence from model 14.0.0 based on crop|
|[rotate]()|60000|10000|28|rotate images(45 and 135) based on crop|

## Performaence
### SVM

|Paramaters| Train Data Set | Performance(Test)|
|:--:      |:--:            |:--:              |
|kernel='rbf'| origin| not converge|
|kernel='linear'| origin| not converge|
|kernel='rbf'| crop| not converge|
|kernel='rbf'| crop2| 92.62%|

### CNN
|Paramaters| Train Data Set | Performance(Test)|
|:--:      |:--:            |:--:              |
|2 convolution, pooling and fc layers| origin| 96.5%|
|2 convolution, pooling and fc layers| crop| 98.25%|
|connect to  SVM| crop| 98.27%|
|kernel='rbf'| crop| not converge|
|kernel='rbf'| crop2| 92.62%|

### Residual Network
|Paramaters| Train Data Set | Performance(Test)|
|:--:      |:--:            |:--:              |
|batch size = 500, epoch = 200| origin| 99.8%|

| Version        | Time           | Diff          | Batch Size|Max Iter|Performance Train |Performance Test |
|:-------------: |:-------------: | :------------:|:---------:|:------:|:----------:|:---:|
| 1.0.0          | 2018-6-5       | Basic Verison |500        |60k     |100%        |97.07%|
| 1.4.0          | 2018-6-12       | img28*28 |1500        |60k     |100%        |98.25%|

## Details
|Model  |Convolution Layers(layers, kernel)     |Pool Layers()                                 |FC Layers|
|:-----:|:---:                                  |:-----:                                       |:----:   |
|1.0.0  |2, [3x3x32][1,1,1,1], [3x3x64][1,1,1,1]|2,(ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1])x2|2|
|1.4.0  |2, [2x2x32][1,1,1,1], [3x3x64][1,1,1,1]|2,(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])x2|2|