load MNIST_all_single.mat 
X=tr_x_255;
Y=tr_y;

%Mdl = fitctree(X,Y)
Mdl= fitctree(X,Y)

view(Mdl)
view(Mdl,'mode','graph')



YPred=predict(Mdl,tr_x_255);
YTest=tr_y;
accuracy = sum(YPred==YTest)/numel(YTest)   %   0.9678


YPred=predict(Mdl,te_x_255(:,:));
YTest=te_y;
accuracy = sum(YPred==YTest)/numel(YTest)    %  0.8782