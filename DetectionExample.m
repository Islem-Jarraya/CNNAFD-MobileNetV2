addpath('Functions\')
%Choose the model to use: Model for horse, model for cat or model for dog
%AnimalModel='Horse';
%AnimalModel='Cat';
AnimalModel='Dog';


%Animal face detection example
%Change the link of the used image
I = imread('Data\Dog-Examples\Dog12.jpg');
[bboxes,scores] = detectYOLOv2CNNAFDImage(I,AnimalModel);
I=imresize(I,[224,224]);
detectedI = insertObjectAnnotation(I,'Rectangle',bboxes,'face');
figure
imshow(detectedI);