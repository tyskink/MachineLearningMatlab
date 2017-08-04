%KNN_1

load MNIST_all_single.mat 
X=tr_x_scale;
Y=tr_y;
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1)


YPred=predict(Mdl,tr_x_scale);
YTest=tr_y;
accuracy = sum(YPred==YTest)/numel(YTest)   %96.40


YPred=predict(Mdl,te_x_scale(:,:));
YTest=te_y;
accuracy = sum(YPred==YTest)/numel(YTest)    %94.45
