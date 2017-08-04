%Liner Classifier for MNIST 

    %from CNN_MNIST_4: 
    
    %20 epoch:  92.09
    
    load MNIST_all_single.mat  
    
    layers = [imageInputLayer([28 28 1],...
            'DataAugmentation',{'none'},...
            'Normalization','none',...                                     zerocenter|none
            'Name','inputlayer')
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
                            'MaxEpochs',20,...              Epochs
                            'MiniBatchSize',128,...         Batch           128     |
                            'Momentum',0.9,...                              0.9     |
                            'Shuffle','once',...                            once    |never
                            'Verbose',1,...                                 1       | 0             — Indicator to display the information on the training progress
                            'VerboseFrequency',100,...                      50      | 0 
                            'OutputFcn',@plotTrainingAccuracy);
convnet = trainNetwork(tr_x4d_scale,tr_y_ca,layers,options);
YPred = classify(convnet,te_x4d_scale);
YTest = te_y_ca;
accuracy = sum(YPred==YTest)/numel(YTest)  

YPred = classify(convnet,tr_x4d_scale);
YTest = tr_y_ca;
accuracy = sum(YPred==YTest)/numel(YTest)