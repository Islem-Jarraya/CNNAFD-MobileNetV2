# CNNAFD-MobileNetV2
The presented code focuses on testing the proposed CNNAFD-MobileNetV2 with YOLOv2 [3] for the face detection problem of three popular animal categories that need control such as horses, cats and dogs. A new Convolutional Neural Network for Animal Face Detection (CNNAFD) is proposed to construct a new backbone CNNAFD-MobileNetV2. CNNAFD used a processed filters based on gradient features and applied with a new way. A new sparse convolutional layer ANOFS-Conv is proposed through a sparse feature selection method known as Automated Negotiation-based Online Feature Selection (ANOFS)[1,2]. The ANOFS method is used as a training optimizer for the new ANOFS-Conv layer. CNNAFD ends by stacked fully connected layers which represent a strong classifier. The fusion of CNNAFD and MobileNetV2[4] constructs the new network CNNAFD-MobileNetV2.

You find the requirements on the Requirements.txt

Dataset: Public datasets (Cat Database, Standford Dogs Dataset, Oxford-IIIT Pet Dataset) and the THDD database.

# References:
[1] BenSaid F and Alimi AM. (2015). ANOFS: Automated negotiation based online feature selection method. International Conference on Intelligent Systems Design and Applications (ISDA). pp 225-230. 10.1109/ISDA.2015.7489229.
[2] BenSaid F and Alimi AM. (2021). Online feature selection system for big data classification based on multi-objective automated negotiation. Pattern Recognition. pp 107-629. 10.1016/j.patcog.2020.107629
[3] Redmon J, Farhadi A. (2017). YOLO9000: Better, Faster, Stronger. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp 6517-6525. 10.1109/CVPR.2017.690
[4] Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp 4510-4520. 10.1109/CVPR.2018.00474
