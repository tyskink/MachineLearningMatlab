this model is trained by MATLAB Function  "SVM_1"



to find out how many sv in every learner:

for index=1:45

 SVnuminLearner(index)=sum(Mdl.BinaryY(:,index)==0);
end
60000-SVnuminLearner'

       12665   // only this is not the same
       11881
       12054
       11765
       11344
       11841
       12188
       11774
       11872
       12700
       12873
       12584
       12163
       12660
       13007
       12593
       12691
       12089
       11800
       11379
       11876
       12223
       11809
       11907
       11973
       11552
       12049
       12396
       11982
       12080
       11263
       11760
       12107
       11693
       11791
       11339
       11686
       11272
       11370
       12183
       11769
       11867
       12116
       12214
       11800
 
sum(Mdl.BinaryY(:,1)==1)   5923
sum(Mdl.BinaryY(:,1)==-1) 6742

sum(Mdl.BinaryLearners{1, 1}.SupportVectorLabels==1)   5923
sum(Mdl.BinaryLearners{1, 1}.SupportVectorLabels==-1)  6609



find out how many vectors are used in the whole system

a=zeros(1,45);

for index=1:60000

 SVnuminLearner(index)=sum(Mdl.BinaryY(index,:)==a);

end

