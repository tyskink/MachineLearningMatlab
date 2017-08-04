%CNN for MNIST 5
    %from CNN_MNIST_4: 
        %   2layer C
        %   
        %   
 
%15 epoch       
%10 epoch     98.00%

testDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\testing', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


trainDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\training', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


layers = [imageInputLayer([28 28 1])
          convolution2dLayer(5,6,...
                                'Stride',1,...
                                'Padding',[0,0])
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(5,12,...
                                'Stride',1,...
                                'Padding',[0,0])
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          
          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer];
      
      
 options = trainingOptions('sgdm',...Environment
                            'CheckpointPath','',...
                            'ExecutionEnvironment','gpu',...                'auto'  | 'cpu' | 'gpu' | 'multi-gpu' | 'parallel'
                            'InitialLearnRate',0.0001,...   Learning Rate
                            'LearnRateSchedule','none',...                  none    |piecewise
                            'LearnRateDropPeriod',10,...
                            'LearnRateDropFactor',0.1,...
                            'L2Regularization',0.0001,...   Regularization
                            'MaxEpochs',10,...              Epochs
                            'MiniBatchSize',128,...         Batch           128     |
                            'Momentum',0.9,...                              0.9     |
                            'Shuffle','once',...                            once    |never
                            'Verbose',1,...                                 1       | 0             � Indicator to display the information on the training progress
                            'VerboseFrequency',100,...                      50      | 0 
                            'OutputFcn',@plotTrainingAccuracy);

convnet = trainNetwork(trainDigitData,layers,options);


YPred = classify(convnet,testDigitData);
YTest = testDigitData.Labels;


accuracy = sum(YPred==YTest)/numel(YTest)