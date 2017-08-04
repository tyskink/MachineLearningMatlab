%CNN for MNIST 6
    %from CNN_MNIST_4: 
        %    2 CCNN Layer
        %    
        
%   'LearnRateSchedule','none'
    %15 epoch       
    %10 epoch    93.28
%   'InitialLearnRate',0.01 'LearnRateSchedule','piecewise'
    % 20 epoch:     99.247(changed rate0.001 at epoch10)
    % 10 epoch:         99.04   98.85   98.68
%   no crosschannelNormalization
    % 10 epoch:             97.3

testDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\testing', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


trainDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\training', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


layers = [imageInputLayer([28 28 1])
                convolution2dLayer([4,3],12,...
                                'Stride',1,...
                                'Padding',[0,0])
                reluLayer
                crossChannelNormalizationLayer(4)
                maxPooling2dLayer(2,'Stride',2)
                convolution2dLayer(5,16)
                reluLayer
                crossChannelNormalizationLayer(4)
                maxPooling2dLayer(2,'Stride',2)
                fullyConnectedLayer(256)
                reluLayer
                fullyConnectedLayer(10)
                softmaxLayer
                classificationLayer];
      
      
 options = trainingOptions('sgdm',...Environment
                            'CheckpointPath','',...
                            'ExecutionEnvironment','gpu',...                'auto'  | 'cpu' | 'gpu' | 'multi-gpu' | 'parallel'
                            'InitialLearnRate',0.01,...   Learning Rate
                            'LearnRateSchedule','piecewise',...                  none    |piecewise
                            'LearnRateDropPeriod',10,...
                            'LearnRateDropFactor',0.1,...
                            'L2Regularization',0.0001,...   Regularization
                            'MaxEpochs',10,...              Epochs
                            'MiniBatchSize',128,...         Batch           128     |
                            'Momentum',0.9,...                              0.9     |
                            'Shuffle','once',...                            once    |never
                            'Verbose',1,...                                 1       | 0             — Indicator to display the information on the training progress
                            'VerboseFrequency',100,...                      50      | 0 
                            'OutputFcn',@plotTrainingAccuracy);

convnet = trainNetwork(trainDigitData,layers,options);


YPred = classify(convnet,testDigitData);
YTest = testDigitData.Labels;


accuracy = sum(YPred==YTest)/numel(YTest)