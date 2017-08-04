%convert MNIST 60000*784 to 4d array

% for index=1:60000
%    
%     tr_4d(:,:,1,index)=(reshape(tr_feats(index,:),28,28))';
%     
% end


% tr_label_ca=categorical(tr_label);
% te_label_ca=categorical(te_label);



for index=1:10000
   
    te_4d(:,:,1,index)=(reshape(te_feats(index,:),28,28))';
    
end
