function score=scoreAFD2(face,w1,w2,w3,w4,w5,w6,w7,net1,net2,net3,net4,net5,net6,net)

face=imresize(face,[305 140]);
[featureVect, hogVisualization] = extractHOGFeatures(face);
res1=w1.*featureVect;
    
    res11=[];
            for i=1:21312
                if (res1(i)==0 && w1(i)==0)
                else
                    res11=[res11,res1(:,i)];
                end
            end
            faux=0;
            res1=res11';
            clear res11;
            y1 = net1(res1);
            %yind1 = round(y1);
            if(y1<0.9)
                faux=faux+1;
                score=y1;
            end
            if(faux<1)
                res2=w2.*featureVect;
                res11=[];
                for i=1:21312
                    if (res2(i)==0 && w2(i)==0)
                    else
                        res11=[res11,res2(:,i)];
                    end
                end
                res2=res11';
                clear res11;
                y2 = net2(res2);
            %yind1 = round(y1);
            if(y2<0.9)
                faux=faux+1;
                score=(y1+y2)/2;
            end
            if(faux<1)
                res3=w3.*featureVect;
                res11=[];
                for i=1:21312
                    if (res3(i)==0 && w3(i)==0)
                    else
                        res11=[res11,res3(:,i)];
                    end
                end
                res3=res11';
                clear res11;
                y3 = net3(res3);
                if(y3<0.9)
                faux=faux+1;
                score=(y1+y2+y3)/3;
                end
                if(faux<1)
                    res4=w4.*featureVect;
                    res11=[];
                    for i=1:21312
                        if (res4(i)==0 && w4(i)==0)
                        else
                            res11=[res11,res4(:,i)];
                        end
                    end
                    res4=res11';
                    clear res11;
                    y4 = net4(res4);
                    if(y4<0.9)
                        faux=faux+1;
                        score=(y4+y3+y2+y1)/4;
                    end
                    if(faux<1)
                        res5=w5.*featureVect;
                        res11=[];
                        for i=1:21312
                            if (res5(i)==0 && w5(i)==0)
                            else
                                res11=[res11,res5(:,i)];
                            end
                        end
                        res5=res11';
                        clear res11;
                        y5 = net5(res5);
                        if(y5<0.9)
                            faux=faux+1;
                            score=(y1+y2+y3+y4+y5)/5;
                        end
                        if(faux<1)
                            res6=w6.*featureVect;
                            res11=[];
                            for i=1:21312
                                if (res6(i)==0 && w6(i)==0)
                                else
                                    res11=[res11,res6(:,i)];
                                end
                            end
                            res6=res11';
                            clear res11;
                            y6 = net6(res6);
                            if(y6<0.9)
                                faux=faux+1;
                                score=(y1+y2+y3+y4+y6)/6;
                            end
                            if(faux<1)
                                res7=w7.*featureVect;
                                res11=[];
                                for i=1:21312
                                    if (res7(i)==0 && w7(i)==0)
                                    else
                                        res11=[res11,res7(:,i)];
                                    end
                                end
                                res7=res11';
                                clear res11;
                                y7 = net(res7);
                                if(y7<0.9)
                                    faux=faux+1;
                                    score=(y1+y2+y3+y4+y5+y6+y7)/7;
                                end
                                if(faux<1)
                                    score=(y1+y2+y3+y4+y5+y6+y7)/7;
                                    
                                end
                            end
                        end
                    end
                     
                end
    
   
            end
            end