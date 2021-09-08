function score=scoreAFD(face,w3,w6,w8,w9,net3,net6,net8,net9)
face=imresize(face,[200 200]);
[featureVect, hogVisualization] = extractHOGFeatures(face);
res1=w3.*featureVect;
    
    res11=[];
            for i=1:20736
                if (res1(i)==0 && w3(i)==0)
                else
                    res11=[res11,res1(:,i)];
                end
            end
            faux=0;
            res1=res11';
            clear res11;
            y3 = net3(res1);
            %yind1 = round(y1);
            score=y3;
            if(y3<0.9)
                faux=faux+1;
                score=y3;
            end
             if(faux<1)
%                 res2=w2.*featureVect;
%                 res11=[];
%                 for i=1:20736
%                     if (res2(i)==0 && w2(i)==0)
%                     else
%                         res11=[res11,res2(:,i)];
%                     end
%                 end
%                 res2=res11';
%                 clear res11;
%                 y2 = net2(res2);
%             %yind1 = round(y1);
%             if(y2<0.9)
%                 faux=faux+1;
%                 score=y2;
%             end
%             if(faux<1)
%                 res3=w3.*featureVect;
%                 res11=[];
%                 for i=1:20736
%                     if (res3(i)==0 && w3(i)==0)
%                     else
%                         res11=[res11,res3(:,i)];
%                     end
%                 end
%                 res3=res11';
%                 clear res11;
%                 y3 = net3(res3);
%                 if(y3<0.9)
%                 faux=faux+1;
%                 score=y3;
%                 end
%                 if(faux<1)
%                     res4=w4.*featureVect;
%                     res11=[];
%                     for i=1:20736
%                         if (res4(i)==0 && w4(i)==0)
%                         else
%                             res11=[res11,res4(:,i)];
%                         end
%                     end
%                     res4=res11';
%                     clear res11;
%                     y4 = net4(res4);
%                     if(y4<0.9)
%                         faux=faux+1;
%                         score=y4;
%                     end
%                     if(faux<1)
%                         res5=w5.*featureVect;
%                         res11=[];
%                         for i=1:20736
%                             if (res5(i)==0 && w5(i)==0)
%                             else
%                                 res11=[res11,res5(:,i)];
%                             end
%                         end
%                         res5=res11';
%                         clear res11;
%                         y5 = net5(res5);
%                         if(y5<0.9)
%                             faux=faux+1;
%                             score=y5;
%                         end
%                         if(faux<1)
                            res6=w6.*featureVect;
                            res11=[];
                            for i=1:20736
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
                                score=(y3+y6)/2;
                            end
                            if(faux<1)
%                                 res7=w7.*featureVect;
%                                 res11=[];
%                                 for i=1:20736
%                                     if (res7(i)==0 && w7(i)==0)
%                                     else
%                                         res11=[res11,res7(:,i)];
%                                     end
%                                 end
%                                 res7=res11';
%                                 clear res11;
%                                 y7 = net7(res7);
%                                 if(y7<0.9)
%                                     faux=faux+1;
%                                     score=y7;
%                                 end
%                                 if(faux<1)
                                    res8=w8.*featureVect;
                                    res11=[];
                                    for i=1:20736
                                        if (res8(i)==0 && w8(i)==0)
                                        else
                                            res11=[res11,res8(:,i)];
                                        end
                                    end
                                    res8=res11';
                                    clear res11;
                                    y8 = net8(res8);
                                    if(y8<0.9)
                                        faux=faux+1;
                                        score=(y3+y6+y8)/3;
                                    end
                                    if(faux<1)
                                        res9=w9.*featureVect;
                                        res11=[];
                                        for i=1:20736
                                            if (res9(i)==0 && w9(i)==0)
                                            else
                                                res11=[res11,res9(:,i)];
                                            end
                                        end
                                        res9=res11';
                                        clear res11;
                                        y9 = net9(res9);
                                        if(y9<0.9)
                                            faux=faux+1;
                                            score=y9;
                                        end
                                        if(faux<1)
                                             score=(y3+y6+y8+y9)/4;
                                             
                                         end
%                                     end
%                                 end
                             end
                      %   end
                    % end
%                      
                 end
%     
%    
             end
            end