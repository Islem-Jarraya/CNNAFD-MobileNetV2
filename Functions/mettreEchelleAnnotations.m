function gthruth = mettreEchelleAnnotations(anno,Database)
gthruth=struct('bboxes',[]);
if strcmp(Database,'THDD')
    chem='Data\THDD-TestSet\';
end
if strcmp(Database,'Cat-Database')
    chem='Data\Cat-Database-TestSet\';
end
if strcmp(Database,'Stanford-Dog-Dataset')
    chem='Data\Stanford-Dogs-Dataset-TestSet\';
end
if strcmp(Database,'Oxford-IIIT-Cat')
    chem='Data\OxfordIIIT-Pet-Dataset-CatTestSet\';
end
if strcmp(Database,'Oxford-IIIT-Dog')
    chem='Data\OxfordIIIT-Pet-Dataset-DogTestSet\';
end

[n k]=size(anno);
for i=1:n
I=imread(strcat(chem,anno.imageFilename{i}));
[H W gg]=size(I);
boxes=anno.face{i};
[a b]=size(boxes);
gthruth(i).bboxes=[];
for j=1:a
gthruth(i).bboxes=[gthruth(i).bboxes;ResizeBox(W,H,boxes(j,:))];
end
end
gthruth(1).bboxes = [gthruth(1).bboxes;1,1,1,1];
gthruth=struct2table(gthruth);
[a b]=size(gthruth.bboxes{1});
gthruth.bboxes{1} = gthruth.bboxes{1}(1:(a-1),:);
end