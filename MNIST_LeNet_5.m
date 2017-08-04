


%matlab for mnist: LeNet

% LeNet5_layers =[ ...
%           imageInputLayer([28 28 1]);
%           convolution2dLayer(5,6,'Stride',1,'Padding',[2,2],'NumChannels',1,'Name','C1');         %(filterSize,numFilters,Name,Value)
%           
%           reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
%           fullyConnectedLayer(10);
%           softmaxLayer();
%           classificationLayer()];


% Example of MatConvNet
% numInputs=28*28;
% numLayers=2;
% biasConnect=[1;0];                  %     numLayers-by-1 Boolean vector           ------If net.biasConnect(i) is 1,       then layer i has a bias, and net.biases{i} is a structure describing that bias.
% inputConnect=ones(1,numInputs);   %     numLayers-by-numInputs Boolean matrix,  ------If net.inputConnect(i,j) is 1,    then layer i has a weight coming from input j, and net.inputWeights{i,j} is a structure describing that weight.
% 
% layerConnect=[0,0;1,0];                 %     numLayer-by-numLayers Boolean vector    ------If net.layerConnect(i,j) is 1,    then layer i has a weight coming from layer j, and net.layerWeights{i,j} is a structure describing that weight.
% outputConnect=[2];                %     1-by-numLayers Boolean vector           ------If net.outputConnect(i) is 1,     then the network has an output from layer i, and net.outputs{i} is a structure describing that output.
% net = network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);



