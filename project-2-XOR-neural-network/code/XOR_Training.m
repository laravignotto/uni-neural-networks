%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script XOR_Training.m
%
% LARA VIGNOTTO, mat 111794
% 20/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Produci un grafico ad ogni run (per grafici punti (4) e (6))
% clear;
% num_run = 5;
% for k = 1:num_run
% figure;
% hold on;
%
alpha = [0.1 0.3 0.5 0.7 0.9];
% alpha = [0.1];
for i = alpha
% %   Prima implementazione
%     vinput = [0 0;
%           0 1;
%           1 0;
%           1 1;
%               ];
%   Seconda implementazione
    vinput = [0 0 1;
            0 1 1;
            1 0 1;
            1 1 1;
                ];
%  
%   I valori attesi in output in corrispondenza di ciascun input  
    correct_output = [0
                    1
                    1
                    0
                    ];

%   Inizializzazione casuale dei pesi
% %   Prima implementazione
%     w0 = 2 * rand(3, 2) - 1;
%     w1 = 2 * rand(1, 3) - 1;
%   Seconda implementazione
    w0 = 2 * rand(3, 3) - 1;
    w1 = 2 * rand(1, 3) - 1;
    w0(3,1) = 0; 
    w0(3,2) = 0; 
    w0(3,3) = 0;
%
% %   Aggiunta di un nodo allo strato nascosto
%     w0 = 2 * rand(4, 3) - 1;
%     w1 = 2 * rand(1, 4) - 1;
%     w0(4,1) = 0; 
%     w0(4,2) = 0; 
%     w0(4,3) = 0;
%
% %   Aggiunta di due nodi allo strato nascosto
%     w0 = 2 * rand(5, 3) - 1;
%     w1 = 2 * rand(1, 5) - 1;
%     w0(5,1) = 0; 
%     w0(5,2) = 0; 
%     w0(5,3) = 0;
%
%   Iperparametro col numero di epoche
    N_epoch = 50000;
%    
%   Inizializzo i vettori per i grafici
    MSE0 = [];          % MSE per epoca
    epoch0 = [];        % epoche
    norm_errors = [];   % norme errori hidden layer
    norm_deltas = [];   % norme delta hidden layer
%
%   Inizia il ciclo di apprendimento sulle epoche
    for epoch = 1:N_epoch
        [w0, w1, final_output, errors_of_hidden_layer, deltas_of_hidden_layer] = XOR_DeltaRule(w0, w1, vinput, correct_output, i);
%
%       Dati per il grafico della curva di apprendimento (loss)
        MSE = immse(final_output, correct_output);  % Mean Squared Error
        MSE0 = [MSE0 MSE];
        epoch0 = [epoch0 epoch];
        norm_errors = [norm_errors norm(errors_of_hidden_layer)];
        norm_deltas = [norm_deltas norm(deltas_of_hidden_layer)];
%
    end
% %   GRAFICI PUNTO (2)
% %   (2) - 1
% %   Utilizzare alpha = [0.1]; e N_epoch = 50000;
% %   Grafico delle curve di apprendimento con diversi alpha
%     plot(epoch0(1:50000),MSE0(1:50000),'LineWidth',2), grid;
%     title('Learning Curve (Loss)')
%     xlabel('epoch')
%     ylabel('MSE')
%     legend('alpha 0.1', 'alpha 0.3', 'alpha 0.5', 'alpha 0.7', 'alpha 0.9');
%     hold on
%
% %   (2) - 2
% %   Utilizzare alpha = [0.1]; e N_epoch = 50000;
% %   Grafico delle norme dell'errore sullo strato nascosto
%     figure;
%     subplot(1,2,1)
%     plot(epoch0(1:50000),norm_errors(1:50000),'LineWidth',2), grid;
%     title('Norme errore su hidden layer')
%     xlabel('epoch')
%     ylabel('Norma error of hidden layers')
%     legend('alpha 0.1');
%     hold on
% %   Grafico delle norme dei delta dello strato nascosto
%     subplot(1,2,2)
%     plot(epoch0(1:50000),norm_deltas(1:50000),'LineWidth',2), grid;
%     title('Norme delta di hidden layer')
%     xlabel('epoch')
%     ylabel('Norma hidden layer delta')
%     legend('alpha 0.1');
%     hold on
%
% %   GRAFICI PUNTI (4) e (6)
% %   Utilizzare alpha = [0.1, 0.3, ...]; e N_epoch = 10000;
%     plot(epoch0(1:20000),MSE0(1:20000),'LineWidth',2), grid;
%     title('Learning Curve (Loss)')
%     xlabel('epoch')
%     ylabel('MSE')
%     legend('alpha 0.1', 'alpha 0.3', 'alpha 0.5', 'alpha 0.7', 'alpha 0.9');
% end
% print(strcat('fig6-4nodi-',int2str(k)),'-depsc') % esporta grafico punti (4) e (6)
% hold off; % per grafici punti (4) e (6)
end
%
% Memorizzazione su file dei valori dei pesi cosi' calcolati
save('XOR_Trained_Network.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%