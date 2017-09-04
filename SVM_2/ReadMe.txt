This model is trained by SVM of svm_lib

for index=1:9
sv_coef_zeronumber(index)=sum(model.sv_coef(:,index)==0);
end

12640	
16056	
14663	
14477	
13393	
12510	
14212	
14357	
12706

19599-sv_coef_zeronumber
6959        
3543        
4936        
5122        
6206        
7089        
5387        
5242        
6893

for indexnum=1:19599
classname(indexnum)=tr_y(model.sv_indices(indexnum));
end

for index=0:9
classnumber(index+1)=sum(classname==index)
end


1097        
1166        
1996        
2285        
1987        
2629        
1450        
1831        
2493        
2665

In the model file, after parameters and other informations such as labels , each line represents a support vector. Support vectors are listed in the order of "labels" shown earlier. (i.e., those from the first class in the "labels" list are grouped first, and so on.) If k is the total number of classes, in front of a support vector in class j, there are k-1 coefficients y*alpha where alpha are dual solution of the following two class problems: 
1 vs j, 2 vs j, ..., j-1 vs j, j vs j+1, j vs j+2, ..., j vs k 
and y=1 in first j-1 coefficients, y=-1 in the remaining k-j coefficients. For example, if there are 4 classes, the file looks like:

+-+-+-+--------------------+
|1|1|1|                    |
|v|v|v|  SVs from class 1  |
|2|3|4|                    |
+-+-+-+--------------------+
|1|2|2|                    |
|v|v|v|  SVs from class 2  |
|2|3|4|                    |
+-+-+-+--------------------+
|1|2|3|                    |
|v|v|v|  SVs from class 3  |
|3|3|4|                    |
+-+-+-+--------------------+
|1|2|3|                    |
|v|v|v|  SVs from class 4  |
|4|4|4|                    |
+-+-+-+--------------------+


extract the coef for each class 

for class=1:10
coef_by_class_5=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
end


class=1;
coef_by_class_5=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_0=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_4=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_1=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_9=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_2=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_3=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_6=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_7=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);
class = class + 1;
coef_by_class_8=model.sv_coef(1+sum(model.nSV(1:class-1)):sum(model.nSV(1:class)),:);


numberofSVforClassifier(1,:)=model.nSV(2)-sum(coef_by_class_0==0);
numberofSVforClassifier(2,:)=model.nSV(4)-sum(coef_by_class_1==0);
numberofSVforClassifier(3,:)=model.nSV(6)-sum(coef_by_class_2==0);
numberofSVforClassifier(4,:)=model.nSV(7)-sum(coef_by_class_3==0);
numberofSVforClassifier(5,:)=model.nSV(3)-sum(coef_by_class_4==0);
numberofSVforClassifier(6,:)=model.nSV(1)-sum(coef_by_class_5==0);
numberofSVforClassifier(7,:)=model.nSV(8)-sum(coef_by_class_6==0);
numberofSVforClassifier(8,:)=model.nSV(7)-sum(coef_by_class_7==0);
numberofSVforClassifier(9,:)=model.nSV(10)-sum(coef_by_class_8==0);
numberofSVforClassifier(10,:)=model.nSV(5)-sum(coef_by_class_9==0);

numberofSVforClassifier(6,:)=model.nSV(1)-sum(coef_by_class_5==0);
numberofSVforClassifier(1,:)=model.nSV(2)-sum(coef_by_class_0==0);
numberofSVforClassifier(5,:)=model.nSV(3)-sum(coef_by_class_4==0);
numberofSVforClassifier(2,:)=model.nSV(4)-sum(coef_by_class_1==0);
numberofSVforClassifier(10,:)=model.nSV(5)-sum(coef_by_class_9==0);
numberofSVforClassifier(3,:)=model.nSV(6)-sum(coef_by_class_2==0);
numberofSVforClassifier(4,:)=model.nSV(7)-sum(coef_by_class_3==0);
numberofSVforClassifier(7,:)=model.nSV(8)-sum(coef_by_class_6==0);
numberofSVforClassifier(8,:)=model.nSV(9)-sum(coef_by_class_7==0);
numberofSVforClassifier(9,:)=model.nSV(10)-sum(coef_by_class_8==0);


570	256	136	333	477	404	439	288	390
392	135	274	311	462	428	276	376	657
680	481	560	466	535	870	741	587	884
1231	397	394	428	666	854	411	562	1074
560	261	273	1271	564	410	477	635	520
579	554	406	687	681	1230	708	497	1081
704	443	475	277	295	740	416	265	517
497	289	625	377	1236	593	555	264	510
1064	395	518	656	750	894	1075	518	509
682	332	1275	316	540	666	289	1250	751


0	570	256	136	333	477	404	439	288	390
392	0	135	274	311	462	428	276	376	657
680	481	0	560	466	535	870	741	587	884
1231	397	394	0	428	666	854	411	562	1074
560	261	273	1271	0	564	410	477	635	520
579	554	406	687	681	0	1230	708	497	1081
704	443	475	277	295	740	0	416	265	517
497	289	625	377	1236	593	555	0	264	510
1064	395	518	656	750	894	1075	518	0	509
682	332	1275	316	540	666	289	1250	751	0



numberofSVforClassifier_extrac=ans

for hang=1:10

for lie=1:10

classifier(hang,lie)=numberofSVforClassifier_extrac(hang,lie)+numberofSVforClassifier_extrac(lie,hang);

end

end






/////////////////////////////////////////////////////////////// 98.46 version
