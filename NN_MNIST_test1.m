

X=tr_feats(1:1000,:);

T=tr_label(1:1000,:);

net = feedforwardnet(25);

[net,tr] = train(net,X',T');