%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function Diabetes_DeltaRule
% Applica la regola di apprendimento
%
% LARA VIGNOTTO, mat 111794
% 15/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [w1, w2, w3, output_matrix] = ...
    Diabetes_DeltaRule(w1, w2, w3, input_data, correct_output, alpha)
%
%%%%%%%%%%%%%%% Settaggio dei parametri
%
    N = 614; % numero dei dati
%
%%%%%%%%%%%%%%% Ciclo sui dati di input
%%%%%%%%%%%%%%%%%%%
    for k = 1:N
%       Trasposta della matrice di input
        input_data_transposed = input_data';
        correct_input_data = input_data_transposed(:,k);
%
%%%%%%%%%%%%%%% Inizia la fase di feedforward
%
%       Trasmissione attivazione dallo strato di input a
%       hidden1
        input_to_hidden_layer1 = w1*correct_input_data;
        output_of_hidden_layer1 = Sigmoid(input_to_hidden_layer1);
%
%       Trasmissione attivazione da hidden1 a hidden2
        input_to_hidden_layer2 = w2*output_of_hidden_layer1;
        output_of_hidden_layer2 = Sigmoid(input_to_hidden_layer2);
%
%       Trasmissione attivazione da hidden2
%       allo strato di output
        input_to_output_node = w3*output_of_hidden_layer2;
%
%       Normalizzazione dei valori di output tramite Sigmoid
        final_output = Sigmoid(input_to_output_node);
%
%       Calcolo della trasposta del correct_output
        correct_output_transpose = correct_output(k, :)';
%
%       Calcolo dell'errore sullo strato di output
        error = correct_output_transpose - final_output;
%
%%%%%%%%%%%%%%% Fase di backpropagation
%       Calcolo della delta rule
        delta = error;
%
        error_of_hidden_layer2 = w3'*delta;
        delta2=(input_to_hidden_layer2>0) .* error_of_hidden_layer2;
%
        error_of_hidden_layer1 = w2'*delta2;
        delta1=(input_to_hidden_layer1>0) .* error_of_hidden_layer1;
%
%       Calcolo dell'aggiornamento dei pesi tramite delta rule
        update_of_w3 = alpha*delta*output_of_hidden_layer2';
        update_of_w2 = alpha*delta2*output_of_hidden_layer1';
        update_of_w1 = alpha*delta1*correct_input_data';
%
%       Aggiornamento delle matrici dei pesi
        w1 = w1 + update_of_w1;
        w2 = w2 + update_of_w2;
        w3 = w3 + update_of_w3;
%
%       Aggiornamento della matrice degli output
        output_matrix(k,:) = final_output';
%
    end % fine ciclo sulle cifre
%
end % fine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%