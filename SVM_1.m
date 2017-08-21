%svm_1




% load MNIST_all_single.mat 
% X=tr_x_scale;
% Y=tr_y;
% 
% SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','gaussian',...
%     'KernelScale','auto');


load MNIST_all_single.mat 
X=tr_x_scale;
Y=tr_y;
t = templateSVM('KernelFunction','gaussian');
Mdl = fitcecoc(X,Y,...
    'Learners',t);


