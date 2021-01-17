%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script Perc_Training.m
% Imposta la procedura di apprendimento di un perceptron monostrato
% Notare che il perceptron monostrato si comporta come una porta logica OR
%
%
% LARA VIGNOTTO, mat 111794
% 13/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = [0.1 0.3 0.5 0.7 0.9];

for i = alpha
%   Calcolo dei vettori di input (training set)
%   In tutto sono quattro, cioè le combinazioni di due bit,
%   sistemati in una matrice 4x2
%   Ogni input vector è una riga della matrice
%     
    input = [0 0;
            0 1;
            1 0;
            1 1;
                ];
%     
%   I valori attesi in output in corrispondenza di ciascun input
%     
    correct_output = [0
                    1
                    1
                    1
                    ];
%   Valore del bias b
    OR_bias = -3;

%   Inizializzazione casuale dei pesi
    OR_Weight = 2 * rand(1, 2) - 1;

%   Iperparametro col numero di epoche
    N_epoch = 1000;

%   Inizializzo i vettori per i grafici
    MSE0 = [];      % MSE per epoch
    epoch0 = [];    % epoch_a
    Weight1 = [];   % Weight per epoch (1)
    Weight2 = [];   % Weight per epoch (2)



%   Inizia il ciclo di apprendimento sulle epoche
    for epoch = 1:N_epoch
%     	Chiama la function che calcola ed esegue la regola
%     	di apprendimento delta rule generalizzata (GDR)
%     	si applica la tecnica di Stochastic Gradient Descent (SGD)
        [OR_Weight,output0] = SGD_DeltaRule_Function(i, OR_Weight, OR_bias, input, correct_output);

%       Dati per il grafico della curva di apprendimento (loss)
        MSE = immse(correct_output, output0');  % Mean Squared Error
        MSE0 = [MSE0 MSE];
        epoch0 = [epoch0 epoch];
        
%       Dati per il grafico dell'evoluzione dei pesi
        Weight1 = [Weight1 OR_Weight(1)];
        Weight2 = [Weight2 OR_Weight(2)];
%
    end

%     SCOMMENTARE PER IL GRAFICO DELLA LOSS CURVE
% %   Figura con il grafico della curva di apprendimento (loss)
%     figure();
%     plot(epoch0(1:300),MSE0(1:300)), grid;
%     title('Learning Curve (Loss)')
%     xlabel('epoch')
%     ylabel('MSE')

%     SCOMMENTARE PER IL GRAFICO DELLA LOSS CURVE DEI PESI
%     % Figura delle curve di apprendimento W1, W2
%     % grafico peso 1
%     figure('Name', 'Curva Loss dei Pesi');
%     subplot(1,2,1);
%     plot(epoch0(1:500),Weight1(1:500)), grid;
%     title('Grafico dei Pesi 1')
%     xlabel('epoch')
%     ylabel('Weight1')
%     % grafico peso 1
%     subplot(1,2,2)
%     plot(epoch0(1:500),Weight2(1:500)), grid;
%     title('Grafico dei Pesi 2')
%     xlabel('epoch')
%     ylabel('Weight2')

%   Grafico delle curve di apprendimento con diversi alpha
    plot(epoch0(1:300),MSE0(1:300)), grid;
    title('Learning Curve (Loss)')
    xlabel('epoch')
    ylabel('MSE')
    legend('alpha 0.1', 'alpha 0.3', 'alpha 0.5', 'alpha 0.7', 'alpha 0.9');
    hold on
end

%
% Memorizzazione su file dei valori dei pesi così calcolati
save('OR_Trained_Network.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%