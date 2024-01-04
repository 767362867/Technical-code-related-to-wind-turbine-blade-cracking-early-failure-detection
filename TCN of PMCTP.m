lgraph = layerGraph();

tempLayers = sequenceInputLayer(75,"Name","input","Normalization","rescale-symmetric");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution1dLayer(5,64,"Name","conv1_1","Padding","causal")
    layerNormalizationLayer("Name","layernorm_1")
    spatialDropoutLayer(0.005)
    convolution1dLayer(5,64,"Name","conv1d","Padding","causal")
    layerNormalizationLayer("Name","layernorm_2")
    reluLayer("Name","relu")
    spatialDropoutLayer(0.005)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution1dLayer(1,64,"Name","convSkip");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution1dLayer(5,64,"Name","conv1_2","DilationFactor",2,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_3")
    spatialDropoutLayer(0.005)
    convolution1dLayer(5,64,"Name","conv1d_1","DilationFactor",2,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_4")
    reluLayer("Name","relu_1")
    spatialDropoutLayer(0.005)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution1dLayer(5,64,"Name","conv1_3","DilationFactor",4,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_5")
    spatialDropoutLayer(0.005)
    convolution1dLayer(5,64,"Name","conv1d_2","DilationFactor",4,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_6")
    reluLayer("Name","relu_2")
    spatialDropoutLayer(0.005)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution1dLayer(5,64,"Name","conv1_4","DilationFactor",8,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_7")
    spatialDropoutLayer(0.005)
    convolution1dLayer(5,64,"Name","conv1d_3","DilationFactor",8,"Padding","causal")
    layerNormalizationLayer("Name","layernorm_8")
    reluLayer("Name","relu_3")
    spatialDropoutLayer(0.005)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
%      fullyConnectedLayer(62,"Name","fc")
     fullyConnectedLayer(1000,"Name","fc1")
    fullyConnectedLayer(75,"Name","fc2")
reluLayer("Name","relu_4")
regressionLayer()];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;


lgraph = connectLayers(lgraph,"input","conv1_1");
lgraph = connectLayers(lgraph,"input","convSkip");
lgraph = connectLayers(lgraph,"convSkip","add_1/in2");
lgraph = connectLayers(lgraph,"layer_2","add_1/in1");
lgraph = connectLayers(lgraph,"add_1","conv1_2");
lgraph = connectLayers(lgraph,"add_1","add_2/in2");
lgraph = connectLayers(lgraph,"layer_4","add_2/in1");
lgraph = connectLayers(lgraph,"add_2","conv1_3");
lgraph = connectLayers(lgraph,"add_2","add_3/in2");
lgraph = connectLayers(lgraph,"layer_6","add_3/in1");
lgraph = connectLayers(lgraph,"add_3","conv1_4");
lgraph = connectLayers(lgraph,"add_3","add_4/in2");
lgraph = connectLayers(lgraph,"layer_8","add_4/in1");


plot(lgraph);
