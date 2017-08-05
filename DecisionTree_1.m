%decision tree
    %according to %KNN_1
    
load MNIST_all_single.mat 
X=tr_x_scale;
Y=tr_y;


Mdl = fitctree(X,Y)

view(Mdl)
view(Mdl,'mode','graph')



YPred=predict(Mdl,tr_x_scale);
YTest=tr_y;
accuracy = sum(YPred==YTest)/numel(YTest)   %   0.9678


YPred=predict(Mdl,te_x_scale(:,:));
YTest=te_y;
accuracy = sum(YPred==YTest)/numel(YTest)    %  0.8782



%notice:
% 1.Test change pixel data into more categories than 0/1:#int(pixel)/128:
% 62.48% divide by 2 is the best

%2. 