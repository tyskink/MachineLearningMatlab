%NN for MNIST 8
    %from CNN_MNIST_7: 
        %    2 NN Layer 800HD
        %    
 
 

testDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\testing', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


trainDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\training', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% layers=convnet.Layers;
layers = [imageInputLayer([28 28 1])
                fullyConnectedLayer(800)
                fullyConnectedLayer(10)
                softmaxLayer
                classificationLayer];
      
      
 options = trainingOptions('sgdm',...Environment
                            'CheckpointPath','',...
                            'ExecutionEnvironment','gpu',...                'auto'  | 'cpu' | 'gpu' | 'multi-gpu' | 'parallel'
                            'InitialLearnRate',0.0001,...   Learning Rate
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