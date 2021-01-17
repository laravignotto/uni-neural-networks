%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script MNIST_Training.m
% Definisce l'architettura della rete;
% Implementa le fasi di apprendimento e di validazione.
%
% VIGNOTTO LARA, mat 111794
% 20/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
[imdsTrain, imdsValidation] = MNIST_DataPrep(0.8);
%
layers = [ ...
    imageInputLayer([28 28 1],'Name','input')

    fullyConnectedLayer(30,'Name','hidden_layer_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu')
   
    fullyConnectedLayer(10,'Name','output_layer')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];
%
%%%%%%%%%%%%%%% Apprendimento
%
% Specifica delle opzioni di apprendimento
options = trainingOptions( ...
    'sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%
% Apprendimento usando il training set
[net,info] = trainNetwork(imdsTrain,layers,options);
%
% Analizza l'architettura di rete
analyzeNetwork(net)
%
%%%%%%%%%%%%%%% Validazione
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
%
% Accuratezza della validazione
accuracy = sum(YPred == YValidation)/numel(YValidation)
%
% Memorizzazione su file
save('MNIST_NeuralNet.mat','info');
