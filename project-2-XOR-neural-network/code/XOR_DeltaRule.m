%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funzione XOR_DeltaRule
% Calcola e applica la Generalized Delta Rule (GDR)
% per implementare una porta logica XOR.
%
% vinput: matrice  dei valori attesi di output in corrispondenza 
% delle singole righe della matrice di input
%
% LARA VIGNOTTO, mat 111794
% 20/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w0, w1, final_output, errors_of_hidden_layer, deltas_of_hidden_layer] = XOR_DeltaRule(w0, w1, vinput, correct_output, alpha)
%
%     alpha = 0.9; % Learning rate
    N = size(vinput,1); % Numero di pattern vector nel training set
    final_output = zeros(N,1);
%   Vettori per il grafico della norma dell'errore sullo strato nascosto
    errors = [];
    deltas = [];
    errors_of_hidden_layer = [];
    deltas_of_hidden_layer = [];
%
%   Calcolo sugli input vector (training set)
    for k = 1:N
        transposed_input = vinput(k, :)';
%
%       Fase di feedforward
        input_to_hidden_layer = w0 * transposed_input;
        output_of_hidden_layer = Sigmoid(input_to_hidden_layer);
%
        input_to_output_node = w1 * output_of_hidden_layer;
        final_output(k) = Sigmoid(input_to_output_node);
%       Calcolo dell'errore sullo strato di output
        final_error = correct_output(k) - final_output(k);
        final_delta = final_output(k) * (1 - final_output(k)) * final_error;
%
%       Fase di backpropagation
        error_of_hidden_layer = w1' * final_delta;
        hidden_layer_delta = SigmoidDerivative(output_of_hidden_layer) .* error_of_hidden_layer;
%       Dati per il grafico della norma dell'errore sullo strato nascosto
        errors = [errors error_of_hidden_layer(1:2)];
        deltas = [deltas hidden_layer_delta(1:2)];
%
%       Calcolo delle correzioni da applicare ai pesi
        dW1 = alpha * final_delta * output_of_hidden_layer';
        dW0 = alpha * hidden_layer_delta * transposed_input';
%       Applicazione delle correzioni (Generalized Delta Rule)
        w0 = w0 + dW0;
        w1 = w1 + dW1;
%       Azzeramento dei pesi ridondanti
        w0(3,1) = 0; 
        w0(3,2) = 0; 
        w0(3,3) = 0;
% %         aggiunta di 1 nodo nascosto
%         w0(4,1) = 0; 
%         w0(4,2) = 0; 
%         w0(4,3) = 0;
% %         aggiunta di 2 nodi nascosti
%         w0(5,1) = 0; 
%         w0(5,2) = 0; 
%         w0(5,3) = 0;
    end 
%   Dati per il grafico della norma dell'errore sullo strato nascosto
    errors_of_hidden_layer = [errors_of_hidden_layer errors];
    deltas_of_hidden_layer = [deltas_of_hidden_layer deltas];
end