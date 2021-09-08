# CNNAFD-MobileNetV2
The presented code focuses on testing the proposed CNNAFD-MobileNetV2 with YOLOv2 for the face detection problem of three popular animal categories that need control such as horses, cats and dogs. A new Convolutional Neural Network for Animal Face Detection (CNNAFD) is proposed to construct a new backbone CNNAFD-MobileNetV2. CNNAFD used a processed filters based on gradient features and applied with a new way. A new sparse convolutional layer ANOFS-Conv is proposed through a sparse feature selection method known as Automated Negotiation-based Online Feature Selection (ANOFS). The ANOFS method is used as a training optimizer for the new ANOFS-Conv layer. CNNAFD ends by stacked fully connected layers which represent a strong classifier. The fusion of CNNAFD and MobileNetV2 constructs the new network CNNAFD-MobileNetV2.

You find the requirements on the Requirements.txt

Dataset: Public datasets (Cat Database, Standford Dogs Dataset, Oxford-IIIT Pet Dataset) and the THDD database.
