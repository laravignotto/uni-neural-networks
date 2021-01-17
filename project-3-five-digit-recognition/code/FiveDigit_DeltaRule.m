%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function FiveDigit_DeltaRule
% Applica la regola di apprendimento
%
% input_image: matrice che rappresenta la cifra in input;
% correct_output: vettore dei valori attesi;
% output_matrix: matrice degli output per ogni cifra;
% w1, w2, w3, w4: matrici dei pesi.
%
% LARA VIGNOTTO, mat 111794
% 03/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [w1, w2, w3, w4, norDelta, norDelta3, norDelta2, norDelta1, output_matrix] = FiveDigit_DeltaRule(w1, w2, w3, w4, input_image, correct_output, alpha, N)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Settaggio dei parametri
%
%     alpha = 0.001; % velocita di apprendimento
%     N = 5; %numero delle cifre da apprendere
    output_matrix = zeros(N,5); % matrice Nx5 (immagine) dei valori di output
    reshaped_input_image = zeros(25,1);
%
%%%%%%%%%%%%%%%%% Ciclo sulle cinque cifre
%%%%%%%%%%%%%%%%%%%
    for k = 1:N
%       Ridimensionamento della matrice dell'immagine di 
%       input in un'unica colonna
        reshaped_input_image = reshape(input_image(:,:,k), 25, 1);
%
%%%%%%%%%%%%%%%%% Inizia la fase di feedforward
%
%       Trasmissione attivazione dallo strato di input a
%       nascosto h1
        input_to_hidden_layer1 = w1*reshaped_input_image;
        output_of_hidden_layer1 = ReLU(input_to_hidden_layer1);
% 
%       Trasmissione attivazione da h1 a h2
        input_to_hidden_layer2 = w2*output_of_hidden_layer1;
        output_of_hidden_layer2 = ReLU(input_to_hidden_layer2);
%
%       Trasmissione attivazione da h2 a h3
        input_to_hidden_layer3 = w3*output_of_hidden_layer2;
        output_of_hidden_layer3 = ReLU(input_to_hidden_layer3);
%
%       Trasmissione attivazione da h3 allo strato di output
        input_to_output_node = w4*output_of_hidden_layer3;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fase di backpropagation
%       Calcolo della delta rule
        delta = error;
        norDelta=norm(delta);
%
        error_of_hidden_layer3 = w4'*delta;
        delta3=(input_to_hidden_layer3>0) .* error_of_hidden_layer3;
        norDelta3 =norm(delta3);
%
        error_of_hidden_layer2 = w3'*delta3;
        delta2=(input_to_hidden_layer2>0) .* error_of_hidden_layer2;
        norDelta2 = norm(delta2);
%
        error_of_hidden_layer1 = w2'*delta2;
        delta1=(input_to_hidden_layer1>0) .* error_of_hidden_layer1;
        norDelta1=norm(delta1);
%
%       Calcolo dell'aggiornamento dei pesi tramite delta rule
        update_of_w4 = alpha*delta*output_of_hidden_layer3';
        update_of_w3 = alpha*delta3*output_of_hidden_layer2'; 
        update_of_w2 = alpha*delta2*output_of_hidden_layer1';
        update_of_w1 = alpha*delta1*reshaped_input_image';
%
%       Aggiornamento delle matrici dei pesi
        w1 = w1 + update_of_w1;
        w2 = w2 + update_of_w2;
        w3 = w3 + update_of_w3;
        w4 = w4 + update_of_w4;
%
%       Aggiornamento della matrice degli output
        output_matrix(k,:) = final_output';
%
    end % fine ciclo sulle cifre
end % fine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%