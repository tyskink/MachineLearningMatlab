%CNN for MNIST  98.x%
testDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\testing', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


trainDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\training', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


layers = [imageInputLayer([28 28 1])
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer];
      
      
options = trainingOptions('sgdm','MaxEpochs',20,...
	'InitialLearnRate',0.0001);

convnet = trainNetwork(trainDigitData,layers,options);


YPred = classify(convnet,testDigitData);
YTest = testDigitData.Labels;


accuracy = sum(YPred==YTest)/numel(YTest)