%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function FiveDigit_Test
% In fase di test della rete, cioè con i valori aggiornati dei pesi,
% tramite una nuova fase di feedforward 
% calcola l'errore rispetto all'output corretto
% w1: matrice 20 x25 dei pesi tra input e primo livello;
% w2: matrice 20 x20 dei pesi tra primo e secondo livelo;
% w3: matrice 20 x20 dei pesi tra secondo e terzo livello;
% w4: matrice 5 x20 dei pesi tra terzo livello e output;
% input_image : matrice 5x5x5 contenente le 5 immagini di input;
% correct_Output : matrice 5x5 dell'output corretto -> riga per riga;
% N : numero di test set
%
% LARA VIGNOTTO, mat 111794
% 12/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MSE_T = FiveDigit_Test (w1, w2, w3, w4, input_image, correct_Output, N)
    learned_Output = zeros([N 5]);
%
    for k = 1:N   % Ciclo sugli stimoli di input
        current_input_image = reshape(input_image(:,:,k),25,1);
%
%       Calcolo dell'output della rete con modalità feedforward
% 
        input_hidden_1 = w1 * current_input_image;
        output_hidden_1 = ReLU(input_hidden_1);
        input_hidden_2 = w2 * output_hidden_1;
        output_hidden_2 = ReLU(input_hidden_2);
        input_hidden_3 = w3 * output_hidden_2;
        output_hidden_3 = ReLU(input_hidden_3);
        input_output_node = w4 * output_hidden_3;
        final_output = Softmax(input_output_node);
%
%       Memorizzazione dell'output finale nella matrice learned_Output
%       dopo aver trasposto il final_output
        learned_Output(k,:) = final_output';
%
    end  % fine ciclo sui vettori (stimoli) di ingresso
%
%   Calcolo l'errore MSE_Test tra le due matrici (output della function)
    MSE_T = immse(learned_Output,correct_Output);
%
end  % fine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
