%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script FiveDigit_Training.m
%
% LARA VIGNOTTO, mat 111794
% 03/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % Produci un grafico ad ogni run ###
% clear;
% num_run = 5;
% for nu = 1:num_run
% figure;
% hold on;
%
alpha = 0.01;
% alpha = [0.001 0.01 0.05]; %(b1)
% densities = [0.1 0.2]; %(c1)
% num_of_elements_of_test_set = [10 50 100]; %(c2)
num_of_elements_of_training_set = [10 15]; %(c3)
%
for i = alpha % inizio ciclo su alpha
% for k = densities %(c1)
% for j = num_of_elements_of_test_set %(c2)
for l = num_of_elements_of_training_set %(c3)
%   input
%   Immagine della cifra uno
    training_set(:,:,1) = [1 0 0 1 1; 1 1 0 1 1; 1 1 0 1 1; 1 1 0 1 1; 1 0 0 0 1];
%   Immagine della cifra due
    training_set(:,:,2) = [0 0 0 0 1; 1 1 1 1 0; 1 0 0 0 1; 0 1 1 1 1; 0 0 0 0 0];
%   Immagine della cifra tre
    training_set(:,:,3) = [0 0 0 0 1; 1 1 1 1 0; 1 0 0 0 1; 1 1 1 1 0; 0 0 0 0 0];
%   Immagine della cifra quattro
    training_set(:,:,4) = [1 1 1 0 1; 1 1 0 0 1; 1 0 1 0 1; 0 0 0 0 0; 1 1 1 0 1];
%   Immagine della cifra cinque
    training_set(:,:,5) = [0 0 0 0 0; 0 1 1 1 1; 0 0 0 0 1; 1 1 1 1 0; 0 0 0 0 1];
%
%   Output corretti: la cifra 'uni' in colonna 'i' indica che si deve individuare la cifra 'i'
    correct_output = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1];
%   Inizializzazione casuale dei pesi
    w1 = 2 * rand(20, 25) - 1;
    w2 = 2 * rand(20, 20) - 1; 
    w3 = 2 * rand(20, 20) - 1;
    w4 = 2 * rand(5, 20) - 1;
%
%   Definizione del test set
%     test_set = FiveDigit_Preparation(training_set,20,k); %(c1)
%     test_set = cat(3,training_set,test_set); %(c1)
%     test_set = FiveDigit_Preparation(training_set,j-5,0.1); %(c2)
%     test_set = cat(3,training_set,test_set); %(c2)
    test_set = FiveDigit_Preparation(training_set,45,0.1); %(c3)
    test_set = cat(3,training_set,test_set); %(c3)
%
%   Iperparametro col numero di epoche
    N_epoch = 1000;
%
%   Inizializzazione dei vettori per i grafici
    MSE_Learn = []; % MSE per epoca (apprendimento)
    MSE_Test = [];  % MSE per epoca (test)
    epoch0 = [];    % epoche
%
%   Settaggio di input_image
%     input_image = training_set; %base
%    training_set_noise = FiveDigit_Preparation(training_set,5,0.1); %(b2)
   training_set_noise = FiveDigit_Preparation(training_set,l-5,0.1); %(c3)
   input_image = cat(3,training_set,training_set_noise); %(b2) e (c3)
%
%   Inizia il ciclo di apprendimento sulle epoche
    for epoch = 1:N_epoch
%         correct_output_training_set = correct_output; %base
%         num_di_training_set = 5; %base
%         correct_output_training_set = repmat(correct_output, 2, 1); %(b2) 
%         num_di_training_set = 10; %(b2)
        correct_output_training_set = repmat(correct_output, l/5, 1); %(c3) 
        num_di_training_set = l; %(c3)
        [w1, w2, w3, w4, norDelta, norDelta3, norDelta2, norDelta1, output_matrix] = FiveDigit_DeltaRule(w1, w2, w3, w4, input_image, correct_output_training_set, i, num_di_training_set);
%
%       Concatena l’epoca corrente al vettore delle epoche
        epoch0 = [epoch0 epoch];
%
%       Concatena l'MSE_Learn osservato in fase di apprendimento sullo 
%       strato di output per l'epoca corrente; calcolato sul training set
        MSE_L = immse(output_matrix, correct_output_training_set);
        MSE_Learn = [MSE_Learn MSE_L];
%
%       Concatena l'MSE_Test osservato dopo l'apprendimento in
%       fase di test; calcolato sul test set
        num_di_test_set = size(test_set,3);
        correct_output_test_set = repmat(correct_output, num_di_test_set/5, 1);
        MSE_T = FiveDigit_Test(w1, w2, w3, w4, test_set, correct_output_test_set, num_di_test_set);
        MSE_Test = [MSE_Test MSE_T];
%
    end % fine ciclo sulle epoche
%
% %   Grafico delle curve di apprendimento con diversi alpha, punti (b1) e (b2)
%     plot(epoch0(1:100),MSE_Learn(1:100),'LineWidth',3), grid;
%     title('Learning Curve (Loss)')
%     xlabel('epoch')
%     ylabel('MSE Learn')
%     legend('alpha 0.001', 'alpha 0.01', 'alpha 0.05');
%     hold on
%
% %   Grafico delle curve loss con diversi test set, punto (c1)
%     plot(epoch0(1:10000),MSE_Test(1:10000),'LineWidth',3), grid;
%     title('Loss Curve con diversi test set')
%     xlabel('epoch')
%     ylabel('MSE Test')
%     leg = legend('dens 0.1', 'dens 0.2');
%     title(leg, 'Test set');
%     hold on
%
% %   Grafico delle curve loss con diversi test set, punto (c2)
%     plot(epoch0(1:1000),MSE_Test(1:1000),'LineWidth',3), grid;
%     title('Loss Curve con diversi test set')
%     xlabel('epoch')
%     ylabel('MSE Test')
%     leg = legend('10 elementi', '50 elementi', '100 elementi');
%     title(leg, 'Test set');
%     hold on
%
%   Grafico delle curve loss con diversi training set, punto (c3)
    plot(epoch0(1:1000),MSE_Test(1:1000),'LineWidth',3), grid;
    title('Loss Curve con diversi training set')
    xlabel('epoch')
    ylabel('MSE Test')
    leg = legend('10 elementi', '15 elementi');
    title(leg, 'Training set');
    hold on
%
% end % ###
% print(strcat('fig-c3-',int2str(nu)),'-depsc') % esporta grafici ###
% hold off; % ###
%
end % (c1) o (c2) o (c3)
end % fine ciclo su alpha
%
% Memorizzazione su file dei valori dei pesi cosi' calcolati
save('FiveDigit_Trained_Network.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%