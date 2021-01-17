%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function MNIST_DataPrep.m
% Accesso al dataset MNIST;
% Definizione dataset locale;
% Splitting.
%
% VIGNOTTO LARA, mat 111794
% 17/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [imds1, imds2] = MNIST_DataPrep(p)
%   Caricamento delle immagini da MNIST
    digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
    imds = imageDatastore(digitDatasetPath,'IncludeSubFolders',true,'LabelSource','foldernames');
%
% %   Visualizzazione di un insieme casuale delle immagini
%     figure
%     perm = randperm(10000,20);
% %
%     for i = 1:20
%         subplot(4, 5, i);
%         imshow(imds.Files{perm(i)});
%     end
%     print(strcat('fig-a3'),'-depsc')
%
%   Splitting
    [imds1,imds2] = splitEachLabel(imds,p,'randomized');
end