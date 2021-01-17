%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script Diabetes_Training.m
%
% LARA VIGNOTTO, mat 111794
% 15/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
alpha = 0.01;
%
[TrainingData, TrainingLabels, ValidationData, ValidationLabels] = ...
    Diabetes_DataPrep();
%
% Inizializzazione casuale dei pesi
w1 = 2 * rand(40, 8) - 1;   % input > hidden1
w2 = 2 * rand(40, 40) - 1;  % hidden1 > hidden2
w3 = 2 * rand(1, 40) - 1;   % hidden2 > output
%
% Iperparametro col numero di epoche
N_epoch = 150;
%
% Inizializzazione dei vettori per i grafici
MSE_Train = []; % MSE per epoca (apprendimento)
epoch0 = [];    % epoche
%
% Inizia il ciclo di apprendimento sulle epoche
for epoch = 1:N_epoch
    [w1, w2, w3, output_matrix] = ...
        Diabetes_DeltaRule(w1, w2, w3, TrainingData, TrainingLabels, alpha);
%
%   Concatena l’epoca corrente al vettore delle epoche
    epoch0 = [epoch0 epoch];
%
%   Concatena l'MSE_Learn osservato in fase di apprendimento sullo 
%   strato di output per l'epoca corrente; calcolato sul training set
    MSE_T = immse(output_matrix, TrainingLabels);
    MSE_Train = [MSE_Train MSE_T];
%
end % fine ciclo sulle epoche
%
%   Grafico della curva di apprendimento 
    plot(epoch0(1:150),MSE_Train(1:150),'LineWidth',3), grid;
    title('Learning Curve (Loss)')
    xlabel('epoch')
    ylabel('MSE Train')
    hold on
%
% Memorizzazione su file dei valori dei pesi cosi' calcolati
save('Diabetes_Trained_Network.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%