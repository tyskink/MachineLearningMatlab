a better version
model=svmtrain(label_train, instance_train,'-s 0 -t 2 -c 4 -g 0.015625');

for index=1:9
sv_coef_zeronumber(index)=sum(model.sv_coef(:,index)==0);
end
12111-sv_coef_zeronumber

3807        
2446        
2857        
2788        
3639        
4242        
3223        
3079        
4047

for indexnum=1:12111
classname(indexnum)=tr_y(model.sv_indices(indexnum));
end

for index=0:9
classnumber(index+1)=sum(classname==index)
end


843         
516        
1419        
1374        
1230        
1466         
948        
1101        
1684        
1530


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

349	218	122	292	351	270	278	210	296
153	77	144	137	179	186	119	164	241
392	299	381	284	318	632	399	397	525
621	292	239	238	416	494	234	314	628
333	238	169	577	363	278	256	401	343
373	275	240	416	411	676	352	276	597
411	303	332	174	212	363	255	185	326
298	195	304	178	548	341	383	171	317
553	314	315	296	498	582	761	316	349
324	235	611	196	343	420	186	645	425


	349	218	122	292	351	270	278	210	296
153		77	144	137	179	186	119	164	241
392	299		381	284	318	632	399	397	525
621	292	239		238	416	494	234	314	628
333	238	169	577		363	278	256	401	343
373	275	240	416	411		676	352	276	597
411	303	332	174	212	363		255	185	326
298	195	304	178	548	341	383		171	317
553	314	315	296	498	582	761	316		349
324	235	611	196	343	420	186	645	425	


