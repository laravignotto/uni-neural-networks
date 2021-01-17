%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script FashionMNIST_Training.m
%
% LARA VIGNOTTO, mat 111794
% 01/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
clear
alpha = 0.01;
[TrainImgs, TrainLabels, ValidationImgs, ValidationLabels] = FashionMNIST_DataPrep();
%
% Inizializzazione casuale dei pesi
w1 = 2 * rand(100, 784) - 1; % input > hidden
w2 = 2 * rand(10, 100) - 1;  % hidden > output
% error('prova')
%
%   Iperparametro col numero di epoche
N_epoch = 2;
%
%   Inizializzazione dei vettori per i grafici
MSE_Train = []; % MSE per epoca (apprendimento)
epoch0 = [];    % epoche
%
% size(TrainImgs)
% Inizia il ciclo di apprendimento sulle epoche
for epoch = 1:N_epoch
    fprintf(['\nEpoca numero >>' num2str(epoch) '\n'])
    [w1, w2, norDelta, norDelta1, output_matrix] = ...
        FashionMNIST_DeltaRule(w1, w2, TrainImgs, TrainLabels, alpha);
%
%   Concatena l’epoca corrente al vettore delle epoche
    epoch0 = [epoch0 epoch];
%
%   Concatena l'MSE_Learn osservato in fase di apprendimento sullo 
%   strato di output per l'epoca corrente; calcolato sul training set
    MSE_T = immse(output_matrix, TrainLabels);
    MSE_Train = [MSE_Train MSE_T];
%
end % fine ciclo sulle epoche
%
%   Grafico della curva di apprendimento 
    plot(epoch0(1:2),MSE_Train(1:2),'LineWidth',3), grid;
    title('Learning Curve (Loss)')
    xlabel('epoch')
    ylabel('MSE Train')
    hold on
% print(strcat('loss-100-epoch-0_01-alpha'),'-depsc')
%
% Memorizzazione su file dei valori dei pesi cosi' calcolati
save('FashionMNIST_Trained_Network.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%