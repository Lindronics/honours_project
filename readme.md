# Level 4 Honours Project - Niklas Lindorfer

![android_build](https://github.com/Lindronics/flir_app/workflows/Android%20CI/badge.svg)
![latex_build](https://github.com/Lindronics/honours_project/workflows/latex_build/badge.svg)

## Deep neural network for classification of multispectral images

### Description

This is the code repository for my 4th year individual project at University of Glasgow. This project was supervised by Professor Roderick Murray-Smith.

The goal of the project was to explore the benefits and drawbacks of adding LWIR images to an image classification model.

### Project abstract

Many conventional image recognition systems are constrained by the limitations of wavelengths visible to the human eye. This project explores the benefits and drawbacks of incorporating thermal images into deep convolutional neural networks for image classification. A multispectral dataset consisting of eight classes of animals and 2317 total samples was captured using a FLIR One Pro thermal camera. Early fusion and late fusion multispectral neural network architectures based on popular image classifiers, such as ResNet, were evaluated on the dataset. The final late fusion model significantly outperforms a comparable visible-light-only system, increasing the validation f1-score from 74% to 80%. The proposed network is unable to beat a state-of-the-art baseline that has been pre-trained on ImageNet and transfer-learned on the custom animals dataset. The final classifier was successfully deployed to an Android mobile application, enabling real-time classification of animals for devices equipped with a FLIR thermal camera.

### Links

* [Android mobile app with TensorFlow lite integration for FLIR image classification](https://github.com/Lindronics/flir_app)

* [Final dissertation PDF](https://1drv.ms/b/s!Aqti0IlhBpFWjocLSKY5smKcBremig)

* [Thermal multispectral dataset download link](https://onedrive.live.com/download?cid=5691066189D062AB&resid=5691066189D062AB%21230121&authkey=ABspHHm_7-RsA8g)
