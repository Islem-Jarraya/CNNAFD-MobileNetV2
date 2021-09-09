function ResultsF = detectYOLOv2CNNAFD(anno,datasetName)
Results=struct('imageFileName',[],'boxes',[],'scores',[]);
ResultsF=struct('imageFileName',[],'boxes',[],'scores',[]);

if strcmp(datasetName,'Stanford-Dog-Dataset')
    load Model/StanfordDogsDatasetModel;
for i=1:100
    imageNameO=anno.imageFilename{i};
    imageName=strcat('Data\Stanford-Dogs-Dataset-TestSet\',imageNameO);
    IO=imread(imageName);
    [H W gg]=size(IO);
    I=imresize(IO,[224 224]);
    resultsyolo=detect(detector,I);
    load('C:\Users\Jarraya\Desktop\islem\horse\boxOut.mat');
    AvantResults=[];
    [a b]=size(boxOut);
    r=0;
    for j=1:a
      if boxOut(j,1)>0 && boxOut(j,2)>0 && boxOut(j,3)>0 && boxOut(j,4)>0 && boxOut(j,5)>0.1
      bbox=[];
      bbox=[round(boxOut(j,1)) round(boxOut(j,2)) round(boxOut(j,3)-boxOut(j,1)) round(boxOut(j,4)-boxOut(j,2))];
      scoreYOLO=boxOut(j,5);
      face=imcrop(IO,ResizeBoxAgrandir(W,H,bbox));
      scoreCNNAFD=scoreAFD(face,w3,w6,w8,w9,net3,net6,net8,net9);
      v=[scoreYOLO scoreCNNAFD];
      scoreFusion=netFusion(v');
        if scoreFusion>0.5 && scoreCNNAFD>0.1
            r=r+1;
            AvantResults(r,1)=bbox(1);
            AvantResults(r,2)=bbox(2);
            AvantResults(r,3)=bbox(3);
            AvantResults(r,4)=bbox(4);
            AvantResults(r,5)=scoreFusion;
            AvantResults(r,6)=scoreYOLO;
            AvantResults(r,7)=scoreCNNAFD;
        end
      end
    end
    Results(i).imageFileName=imageNameO;
    label=[];
    label(1:r)=1;
    if r>0
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(AvantResults(:,1:4),AvantResults(:,5),label','OverlapThreshold',0.3);

    Results(i).boxes=bboxes;
    Results(i).scores=scores;
    else

    end
end
for i=1:100
    ResultsF(i).imageFileName=Results(i).imageFileName;
    [a b]=size(Results(i).boxes);
    bboxes=[];
    sscores=[];
    score=Results(i).scores;
    bbox=Results(i).boxes;
    for j=1:a
      if(score(j)>0.7)
         bboxes=[bboxes;bbox(j,:)];
         sscores=[sscores;score(j)];
      end
    end
    ResultsF(i).boxes=bboxes;
    ResultsF(i).scores=sscores;
end
end
if strcmp(datasetName,'Cat-Database')
load Model/CatDatabaseModel;
for i=1:3000
    imageNameO=anno.imageFilename{i};
    imageName=strcat('Data\Cat-Database-TestSet\',imageNameO);
    IO=imread(imageName);
    [H W gg]=size(IO);
    I=imresize(IO,[224 224]);
    resultsyolo=detect(detector,I);
    load('C:\Users\Jarraya\Desktop\islem\horse\boxOut.mat');
    AvantResults=[];
    [a b]=size(boxOut);
    r=0;
    for j=1:a
      if boxOut(j,1)>0 && boxOut(j,2)>0 && boxOut(j,3)>0 && boxOut(j,4)>0 && boxOut(j,5)>0.1
      bbox=[];
      bbox=[round(boxOut(j,1)) round(boxOut(j,2)) round(boxOut(j,3)-boxOut(j,1)) round(boxOut(j,4)-boxOut(j,2))];
      scoreYOLO=boxOut(j,5);
      face=imcrop(IO,ResizeBoxAgrandir(W,H,bbox));
      scoreCNNAFD=scoreAFD2(face,w1,w2,w3,w4,w5,w6,w7,w8,w9,net1,net2,net3,net4,net5,net6,net7,net8,net9);
      if scoreCNNAFD>0.1
      v=[scoreYOLO scoreCNNAFD];
      scoreFusion=netFusion(v');
        if scoreFusion>0.5
            r=r+1;
            AvantResults(r,1)=bbox(1);
            AvantResults(r,2)=bbox(2);
            AvantResults(r,3)=bbox(3);
            AvantResults(r,4)=bbox(4);
            AvantResults(r,5)=scoreFusion;
            AvantResults(r,6)=scoreYOLO;
            AvantResults(r,7)=scoreCNNAFD;
        end
      end
      end
    end
    Results(i).imageFileName=imageNameO;
    label=[];
    label(1:r)=1;
    if r>0
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(AvantResults(:,1:4),AvantResults(:,5),label','OverlapThreshold',0.3);
    Results(i).boxes=bboxes;
    Results(i).scores=scores;
    end
end
for i=1:3000
    ResultsF(i).imageFileName=Results(i).imageFileName;
    [a b]=size(Results(i).boxes);
    bboxes=[];
    sscores=[];
    score=Results(i).scores;
    bbox=Results(i).boxes;
    for j=1:a
      if(score(j)>0.7)
         bboxes=[bboxes;bbox(j,:)];
         sscores=[sscores;score(j)];
      end
    end
    ResultsF(i).boxes=bboxes;
    ResultsF(i).scores=sscores;
end
end
if strcmp(datasetName,'Oxford-IIIT-Cat')
load Model/CatDatabaseModel;
for i=1:1188
    imageNameO=anno.imageFilename{i};
    imageName=strcat('Data\OxfordIIIT-Pet-Dataset-CatTestSet\',imageNameO);
    IO=imread(imageName);
    [H W gg]=size(IO);
 else
     I=imresize(IO,[224 224]);
    resultsyolo=detect(detector,I);
    load('C:\Users\Jarraya\Desktop\islem\horse\boxOut.mat');
    AvantResults=[];
    [a b]=size(boxOut);
    r=0;
    for j=1:a
      if boxOut(j,1)>0 && boxOut(j,2)>0 && boxOut(j,3)>0 && boxOut(j,4)>0 && boxOut(j,5)>0.1
      bbox=[];
      bbox=[round(boxOut(j,1)) round(boxOut(j,2)) round(boxOut(j,3)-boxOut(j,1)) round(boxOut(j,4)-boxOut(j,2))];
      %ProposalsHorses(k).Boxes=bbox;
      scoreYOLO=boxOut(j,5);
      face=imcrop(IO,ResizeBoxAgrandir(W,H,bbox));
      scoreCNNAFD=scoreAFD2(face,w1,w2,w3,w4,w5,w6,w7,w8,w9,net1,net2,net3,net4,net5,net6,net7,net8,net9);
      if scoreCNNAFD>0.1
      v=[scoreYOLO scoreCNNAFD];
      scoreFusion=netFusion(v');
        if scoreFusion>0.5
            r=r+1;
            AvantResults(r,1)=bbox(1);
            AvantResults(r,2)=bbox(2);
            AvantResults(r,3)=bbox(3);
            AvantResults(r,4)=bbox(4);
            AvantResults(r,5)=scoreFusion;
            AvantResults(r,6)=scoreYOLO;
            AvantResults(r,7)=scoreCNNAFD;
        end
      end
      end
    end
    Results(i).imageFileName=imageNameO;
    label=[];
    label(1:r)=1;
    if r>0
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(AvantResults(:,1:4),AvantResults(:,5),label','OverlapThreshold',0.3);
    Results(i).boxes=bboxes;
    Results(i).scores=scores;
    end
end
for i=1:1188
    ResultsF(i).imageFileName=Results(i).imageFileName;
    [a b]=size(Results(i).boxes);
    bboxes=[];
    sscores=[];
    score=Results(i).scores;
    bbox=Results(i).boxes;
    for j=1:a
      if(score(j)>0.7)
         bboxes=[bboxes;bbox(j,:)];
         sscores=[sscores;score(j)];
      end
    end
    ResultsF(i).boxes=bboxes;
    ResultsF(i).scores=sscores;

end
end
if strcmp(datasetName,'THDD')
    load Model/THDDModel;
for i=1:400
    imageNameO=anno.imageFilename{i};
    imageName=strcat('Data\THDD-TestSet\',imageNameO);
    IO=imread(imageName);
    [H W gg]=size(IO);
    I=imresize(IO,[224 224]);
    resultsyolo=detect(detector,I);
    load('C:\Users\Jarraya\Desktop\islem\horse\boxOut.mat');
    AvantResults=[];
    [a b]=size(boxOut);
    r=0;
    for j=1:a
      if boxOut(j,1)>0 && boxOut(j,2)>0 && boxOut(j,3)>0 && boxOut(j,4)>0 && boxOut(j,5)>0.1
      bbox=[];
      bbox=[round(boxOut(j,1)) round(boxOut(j,2)) round(boxOut(j,3)-boxOut(j,1)) round(boxOut(j,4)-boxOut(j,2))];
      
      scoreYOLO=boxOut(j,5);
      face=imcrop(IO,ResizeBoxAgrandir(W,H,bbox));
      scoreCNNAFD=scoreAFD3(face,w1,w2,w3,w4,w5,w6,w7,net1,net2,net3,net4,net5,net6,net);
      
      v=[scoreYOLO scoreCNNAFD];
      scoreFusion=netFusion(v');
        if scoreFusion>0.99999
            r=r+1;
            AvantResults(r,1)=bbox(1);
            AvantResults(r,2)=bbox(2);
            AvantResults(r,3)=bbox(3);
            AvantResults(r,4)=bbox(4);
            AvantResults(r,5)=scoreFusion;
            AvantResults(r,6)=scoreYOLO;
            AvantResults(r,7)=scoreCNNAFD;
            if scoreYOLO<0.0001 || scoreCNNAFD<0.0001
               AvantResults(r,6)=0;
               AvantResults(r,7)=0;
            end
            AvantResults(r,8)=(scoreYOLO+scoreCNNAFD)/2;
        end
      end
    end
    Results(i).imageFileName=imageNameO;
    label=[];
    label(1:r)=1;
    if r>0
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(AvantResults(:,1:4),AvantResults(:,8),label','OverlapThreshold',0.4);
    [abb bbb]=size(bboxes);
    for faraj=1:abb
        for faraj2=1:r
          if(bboxes(faraj,1)==AvantResults(faraj2,1) && bboxes(faraj,2)==AvantResults(faraj2,2) && bboxes(faraj,3)==AvantResults(faraj2,3) && bboxes(faraj,4)==AvantResults(faraj2,4))
              scores(faraj)=AvantResults(faraj2,5);
          end
        end
    end
    Results(i).boxes=bboxes;
    Results(i).scores=scores;
    else

    end
 
end
for i=1:400
    ResultsF(i).imageFileName=Results(i).imageFileName;
    [a b]=size(Results(i).boxes);
    bboxes=[];
    sscores=[];
    score=Results(i).scores;
    bbox=Results(i).boxes;
    for j=1:a
      if(score(j)>0.7)
         bboxes=[bboxes;bbox(j,:)];
         sscores=[sscores;score(j)];
      end
    end
    ResultsF(i).boxes=bboxes;
    ResultsF(i).scores=sscores;
end
end
end
