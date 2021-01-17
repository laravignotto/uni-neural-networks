%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function FashionMNIST_DeltaRule
% Applica la regola di apprendimento
%
% LARA VIGNOTTO, mat 111794
% 01/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [w1, w2, norDelta, norDelta1, output_matrix] = ...
    FashionMNIST_DeltaRule(w1, w2, input_image, correct_output, alpha)
%
%%%%%%%%%%%%%%% Settaggio dei parametri
%
    N = 56000; % numero delle labels
%
%%%%%%%%%%%%%%% Ciclo sulle immagini di input
%%%%%%%%%%%%%%%%%%%
    for k = 1:N
%       Ridimensionamento della matrice dell'immagine di 
%       input in un'unica colonna
        reshaped_input_image = reshape(input_image(:,:,k), 784, 1);
%
%%%%%%%%%%%%%%% Inizia la fase di feedforward
%
%       Trasmissione attivazione dallo strato di input a
%       nascosto
        input_to_hidden_layer = w1*reshaped_input_image;
        output_of_hidden_layer = ReLU(input_to_hidden_layer);
%
%       Trasmissione attivazione dallo strato nascosto
%       allo strato di output
        input_to_output_node = w2*output_of_hidden_layer;
%
%       Normalizzazione dei valori di output tramite Softmax
        final_output = Softmax(input_to_output_node);
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
        norDelta=norm(delta);
%
        error_of_hidden_layer = w2'*delta;
        delta1=(input_to_hidden_layer>0) .* error_of_hidden_layer;
        norDelta1=norm(delta1);
%
%       Calcolo dell'aggiornamento dei pesi tramite delta rule
        update_of_w2 = alpha*delta*output_of_hidden_layer';
        update_of_w1 = alpha*delta1*reshaped_input_image';
%
%       Aggiornamento delle matrici dei pesi
        w1 = w1 + update_of_w1;
        w2 = w2 + update_of_w2;
%
%       Aggiornamento della matrice degli output
        output_matrix(k,:) = final_output';
%
    end % fine ciclo sulle cifre
%
end % fine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%