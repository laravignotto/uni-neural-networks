%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funzione SGD_DeltaRule_function
% Calcola e applica la Generalized Delta Rule (GDR)
% 
% OR_Weight: matrice dei pesi (1 x 2)
% bias: variabile di contesto
% input: matrice dei valori di input per l'apprendimento
% correct_output: vettore dei valori attesi di output in corrispondenza
% delle singole righe della matrice di input
%
% LARA VIGNOTTO, mat 111794
% 13/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OR_Weight,output0] = SGD_DeltaRule_Function(alpha, OR_Weight, OR_bias, input, correct_output)
%
% 	Iperparametro per regolare la velocità di apprendimento (learning grade)
% 	alpha = 0.8;
% 	Numero dei vettori di input, quattro nel caso presente
	N = size(input, 1);
    output0 = [];
%
	for k = 1:N 
% 		Una singola combinazione dei valori di input, cioè il vettore di input,
% 		è una riga: è necessario trasporla in colonna
		transposed_input = input(k, :)';
%
%		Valore corretto, noto a-priori
		corval = correct_output(k);
%
% 		Calcolo dell'input al nodo di uscita
		weighted_sum = OR_Weight * transposed_input + OR_bias;
%
% 		Calcolo dell'output con la funzione di attivazione sigmoide
		output = Sigmoid(weighted_sum);
        output0 = [output0 output];
%
%		Calcolo dell'errore in output
		error = corval - output;
%
%		Calcolo del delta con la Generalized Delta Rule (GDR)
		delta = output * (1 - output) * error;
%
%		Calcolo della correzione (aggiornamento) dei pesi con la GDR
		dWeight = alpha * delta * transposed_input;
%
%		Applicazione della GDR
		OR_Weight = OR_Weight + dWeight';
%
	end	% fine ciclo
%
end	% fine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

