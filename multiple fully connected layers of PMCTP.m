XTest=[ResNet50.errTest(:,1) ResNet50.errTest(:,1) ResNet50.errTest(:,1) Xception.errTest(:,1) ResNet50TCN.errTest(:,1) ResNet50TCN.errTest(:,1) ResNet50TCN.errTest(:,1) XceptionTCN.errTest(:,1)];
layers = [
                    featureInputLayer(8,"Name","featureinput")
                    fullyConnectedLayer(200,"Name","fc_2")
                    %reluLayer("Name","relu_1")
                    %dropoutLayer(0.5,"Name","dropout_1")
                    fullyConnectedLayer(200,"Name","fc_3")
                    %reluLayer("Name","relu_2")
                    %dropoutLayer(0.5,"Name","dropout_2")
                    fullyConnectedLayer(10,"Name","fc_4")
                    %reluLayer("Name","relu_3")激活函数
                    %dropoutLayer(0.5,"Name","dropout_3")
                    fullyConnectedLayer(2,"Name","fc_5")
                    softmaxLayer("Name","softmax")
                    classificationLayer("Name","classoutput")];
                opts = trainingOptions("sgdm",...
                    "InitialLearnRate",0.01,...
                    "MaxEpochs",300,...
                    "MiniBatchSize",954,...
                    "Shuffle","every-epoch",...
                    "Plots","none",...
                    "VerboseFrequency",20,...
                    'ExecutionEnvironment','gpu');
                tic
                [net, traininfo] = trainNetwork(XTest,YTest,layers,opts);
                usedtime=toc
                [YPredTest,errTest] = classify(net,XTest);
                accTest = sum(YPredTest == YTest)./numel(YTest)
                TN=sum(YPredTest(1:sum(YTest=='0'),:)=='0');   %真实为0，预测也为0
                FP=sum(YPredTest(1:sum(YTest=='0'),:)=='1');
                TP=sum(YPredTest(sum(YTest=='0')+1:numel(YTest),:)=='1');
                FN=sum(YPredTest(sum(YTest=='0')+1:numel(YTest),:)=='0');
                Acc=(TP+TN)/(TN+TP+FP+FN)
                 P=TP/(TP+FP)
                R=TP/(TP+FN)
                F1=2*P*R/(P+R)