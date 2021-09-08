%CNNAFD-MobileNetV2 detection
%You can use four trained models; a model trained on THDD, a model trained
%on Cat Database and a model trained on Stanford Dog Datase.
%You can chose the tested data. Only write the name of the dataset on the detectYOLOv2CNNAFDDog function parameters.

addpath('Functions\')
%Test THDD
Database='THDD';

%Test Cat-Database.
%Database='Cat-Database';

%Test Oxford-IIIT Pet Dataset (cat set).
%Database='Oxford-IIIT-Cat';

%Test Stanford-Dog-Dataset.
%Database='Stanford-Dog-Dataset';

%Test Oxford-IIIT Pet Dataset (dog set).
%Database='Oxford-IIIT-Dog';

Results = detectYOLOv2CNNAFD(anno,Database);
Results=struct2table(Results);
%Ground-truth Extraction
gthruth = mettreEchelleAnnotations(anno,Database);
%Detection Evaluation
[averagePrecision,recall,precision] = evaluateDetectionPrecision(Results(:,2:3),gthruth);
fprintf('AP:%f\n',averagePrecision*100);
[precision,recall] = bboxPrecisionRecall(Results(:,2),gthruth);
fprintf('P: %f, R: %f\n',precision*100,recall*100);