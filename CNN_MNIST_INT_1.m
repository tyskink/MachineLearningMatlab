% CNN_MNIST_INT_1
% test by using int parameters
%from CNN_MNIST_4: 

load C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\MNIST_all_single.mat  


layers = [imageInputLayer([28 28 1],...
            'DataAugmentation',{'none'},...
            'Normalization','zerocenter',...                                     zerocenter|none
            'Name','inputlayer')
          convolution2dLayer(5,6,...
                                'Stride',1,...
                                'Padding',[0,0],...
                                'name','C1')
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          
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

pause;

%   get the zero-center parameters
    im=zeros(28,28)
    im4d(:,:,1,1)=im
    features = activations(convnet,im4d,1)
    im=(reshape(features,28,28))
    zerocenter=im
    zerocenter=int32(zerocenter*255)
    
    file=fopen('C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\Zc.lki','w')
    fwrite(file,zerocenter','int32')
    fclose(file)
    
%	save kernel on C1    
for index=1:6
    C1=int32(convnet.Layers(2,1).Weights(:,:,1,index)*255)
    file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\C1','K',num2str(index),'.lki'],'w')
    fwrite(file,C1','int32')
    fclose(file)
end    
    
%save bias on C1
for index=1:6
    C1B(index)=int32(convnet.Layers(2,1).Bias(1,1,index)*255)
end
file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\C1B.lki'],'w')
fwrite(file,C1B','int32')
fclose(file)    
    
%convert the matlab FC matrix to C order
F5W=zeros(10,784);
i=1;
    for IM=1:6
        for IL=1:12
            for IC=1:12
                F5W(:,(IM-1)*144+IL+(IC-1)*12)=convnet.Layers(5,1).Weights(:,i);
                i=i+1;
            end
        end       
    end
% Save to File
 
F5W=int32(F5W*255);
file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\F5W.lki'],'w')
fwrite(file,F5W','int32')
fclose(file)
F5B=int32(convnet.Layers(5,1).Bias*255);
file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\F5B.lki'],'w')
fwrite(file,F5B','int32')
fclose(file)
    
    