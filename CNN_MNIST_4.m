%CNN for MNIST 4
    %from CNN_MNIST_3: 
        %   Padding on C1:  2->0
        %   Filters on C1:  20->10
 
%15 epoch       98.31   98.34
%10 epoch   98.15   


    % Filter on C1 ==6
        %10 epoch   97.84
       
    % 'InitialLearnRate',0.01,...
        %10 epoch   NaN
        
    % 'InitialLearnRate',0.001,...'LearnRateSchedule','piecewise',...  
        %5  epoch   90
        %10 epoch
        
        
    % using the scale data
        %10 epoch   95              94.53
        %20 epoch       95.03
        
        
    %none zero center in image input layer
        %10 epoch           91.92
        
    % 'InitialLearnRate',0.0001,...'    + zero -center
        %10 epoch   84.63
        %20 epoch   86.52
        
    %using 4d none-scale data + 0.001 rate
        %20 epoch   97.56
        
    %using 4d-scale data +0.01 rate
        %20 epoch   98.43   98.38   98.38
    
    %using single input
        %20 epoch   98.08   98.09   98.09   98.74
%load C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\MNIST_LK.mat
load C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\MNIST_all_single.mat        
        
% testDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\testing', ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% 
% trainDigitData = imageDatastore('C:\Users\kongq\Desktop\machine_learning_ex\DataSet\MNIST\mnist_png\training', ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');


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

%convnet = trainNetwork(trainDigitData,layers,options);
% YPred = classify(convnet,testDigitData);
% YTest = testDigitData.Labels;
% accuracy = sum(YPred==YTest)/numel(YTest)

convnet = trainNetwork(tr_x4d_scale,tr_y_ca,layers,options);
YPred = classify(convnet,te_x4d_scale);
YTest = te_y_ca;
accuracy = sum(YPred==YTest)/numel(YTest)

 
YPred = classify(convnet,tr_x4d_scale);
YTest = tr_y_ca;
accuracy = sum(YPred==YTest)/numel(YTest)

% convnet = trainNetwork(tr_x4d,tr_y_ca,layers,options);
% YPred = classify(convnet,te_x4d);
% YTest = te_y_ca;
% accuracy = sum(YPred==YTest)/numel(YTest)

% see the features on every layer
    % features = activations(convnet,te_x4d_scale(:,:,1,1),1)%ImageInput
    % im=(reshape(features,28,28))
    % imshow(im)

%     features = activations(convnet,te_x4d_scale(:,:,1,1),2)%Conv
%     im=(reshape(features,24,24))
%     imshow(im)

%     features = activations(convnet,te_x4d_scale(:,:,1,1),3) %ReLu
%     im=(reshape(features,24,24))
%     imshow(im)

%     features = activations(convnet,te_x4d_scale(:,:,1,1),4) %MaxPolling
%     im=(reshape(features,12,12,6))
%     imshow(im(:,:,1))
    
%     features = activations(convnet,te_x4d_scale(:,:,1,1),5) %FC
%     im=(reshape(features,24,24))
%     imshow(im)    
%    features'-convnet.Layers(5,1).Bias

%   features = activations(convnet,te_x4d_scale(:,:,1,1),6) %SoftMax
    
% find out a specific value
    % fa=features((features>0.22263))
    % fb=fa((fa<0.22264))

    % tr_x4d_scale(:,:,1,1)-mean(mean(tr_x4d_scale(:,:,1,1)))


%get the zero-center parameters
%     im=zeros(28,28)
%     im4d(:,:,1,1)=im
%     features = activations(convnet,im4d,1)
%     im=(reshape(features,28,28))
%     zerocenter=im
    
    %test
    %features = activations(convnet,te_x4d_scale(:,:,1,1),1)
    %features_1=(reshape(features,28,28))
    %features_2=te_x4d_scale(:,:,1,1)-zerocenter
    %features_21=te_x4d_scale(:,:,1,1)+zerocenter  %is correct
    
%     zerocenter_double=double(zerocenter)
%     file=fopen('C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\Zc.lkd','w')
%     fwrite(file,zerocenter_double','double')
%     fclose(file)
    
%save kernel on C1
% for index=1:6
%     C1=double(convnet.Layers(2,1).Weights(:,:,1,index))
%     file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\C1','K',num2str(index),'.lkd'],'w')
%     fwrite(file,C1','double')
%     fclose(file)
% end

%save bias on C1
% for index=1:6
%     C1B(index)=double(convnet.Layers(2,1).Bias(1,1,index))
% end
% file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\C1B.lkd'],'w')
% fwrite(file,C1B','double')
% fclose(file)


%convert the matlab FC matrix to C order
% F5W=zeros(10,784);
% i=1;
%     for IM=1:6
%         for IL=1:12
%             for IC=1:12
%                 F5W(:,(IM-1)*144+IL+(IC-1)*12)=convnet.Layers(5,1).Weights(:,i);
%                 i=i+1;
%             end
%         end       
%     end

% for i=1:864
%  F5W(:,A(i))=convnet.Layers(5,1).Weights(:,i);
% end

%Test the Correct Order
%     features = activations(convnet,te_x4d_scale(:,:,1,1),4); %MaxPolling     1*864
%     convnet.Layers(5,1).Weights*features'
%     activations(convnet,te_x4d_scale(:,:,1,1),5) %FC
    %this result is correct
    
    %the order in C: features
%     features = activations(convnet,te_x4d_scale(:,:,1,1),4); %MaxPolling     1*864
%     im=(reshape(features,12,12,6));%the same as the result of C
%     for index=1:6
%     Corder(1+(index-1)*144:144+(index-1)*144)=reshape(im(:,:,index)',[],1);
%     end
%   F5W*Corder' %Correct!
    
% Save to File
%   F5W=convnet.Layers(5,1).Weights;
% F5W=double(F5W);
% file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\F5W.lkd'],'w')
% fwrite(file,F5W','double')
% fclose(file)

% F5B=double(convnet.Layers(5,1).Bias);
% file=fopen(['C:\Users\kongq\Desktop\machine_learning_ex\CNN_ZcCoReSuFuSm\F5B.lkd'],'w')
% fwrite(file,F5B','double')
% fclose(file)
