%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function FashionMNIST_DataPrep.m
% Accesso al dataset Fashion MNIST;
% Definizione dataset locale (training e validation set).
%
% VIGNOTTO LARA, mat 111794
% 27/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [TrainImgs, TrainLabels, ValidationImgs, ValidationLabels] = FashionMNIST_DataPrep()
%   Caricamento delle immagini da Fashion MNIST
    filenameImagesTrain = 'fashion/train-images-idx3-ubyte';
    filenameLabelsTrain = 'fashion/train-labels-idx1-ubyte';
    filenameImagesTest = 'fashion/t10k-images-idx3-ubyte';
    filenameLabelsTest = 'fashion/t10k-labels-idx1-ubyte';

    XTrain = processImagesFashionMNIST(filenameImagesTrain);
    YTrain = processLabelsFashionMNIST(filenameLabelsTrain);
    XTest = processImagesFashionMNIST(filenameImagesTest);
    YTest = processLabelsFashionMNIST(filenameLabelsTest);
%
% %   Visualizzazione di un insieme casuale delle immagini
%     figure
%     perm = randperm(60000,20);
% %
%     for i = 1:20
%         subplot(4, 5, i);
%         images = extractdata(XTrain(:,:,1,:));
%         imshow(images(:,:,1,perm(i)));
%     end
%     print(strcat('20-immagini-random'),'-depsc')
%    
%%%%%%%%%%%%%%%%%%% Raggruppamento
%   Numero totale delle immagini contenute negli archivi
    numtotImages = 70000;
%  
%   Inizializzazione degli array
    images = zeros(28,28,numtotImages);
    labels = zeros(numtotImages, 1);
%
%	Raggruppamento delle immagini delle etichette
    images(:,:,1:60000) = extractdata(XTrain);
    images(:,:,60001:numtotImages) = extractdata(XTest);
    labels(1:60000,:) = YTrain;
	labels(60001:numtotImages,:) = YTest;
%
%	La variabile labels è convertita in forma matriciale
%	tramite la funzione label2matrix.   
	labels = label2matrix(labels);
%   
%%%%%%%%%%%%%%%%%%% Splitting
%	Percentuale di splitting
	training_perc = 0.8; 
%	Randomizzazione degli indici delle immagini
	split_perm = randperm(numtotImages);
%	Cardinalità dell’insieme di apprendimento (training)
	training_cardin = floor(numtotImages * training_perc);

%	Cardinalità dell’insieme di collaudo (validation)
    validation_cardin = numtotImages - training_cardin;   

%	Definizione degli insiemi di apprendimento 
%	e delle relative etichette, tutti randomizzati
	TrainImgs=images(:,:,split_perm(1:training_cardin));
    TrainLabels=labels(split_perm(1:training_cardin),:);
    
%	Definizione degli insiemi di collaudo 
%	e delle relative etichette, tutti randomizzati
	ValidationImgs = ...
        images(:,:,split_perm(training_cardin+1:end));
	ValidationLabels = ... 
        labels(split_perm(training_cardin+1:end),:);
%
end