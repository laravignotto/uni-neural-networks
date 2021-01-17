%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script XOR_Test.m per applicare la procedura di apprendimento
% di un perceptron multistrato, 
% che in questo caso si comporta come una porta logica OR
%
% LARA VIGNOTTO, mat 111794
% 20/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Carica i dati della rete gia' addestrata
load('XOR_Trained_Network.mat')
%
transposed_input = vinput(1, :)';
input_to_hidden_layer = w0 * transposed_input;
output_of_hidden_layer = Sigmoid(input_to_hidden_layer);
%
input_to_output_node = w1 * output_of_hidden_layer;
final_output(1) = Sigmoid(input_to_output_node)