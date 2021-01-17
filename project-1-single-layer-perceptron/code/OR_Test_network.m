%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script OR_Test_network.m per applicare la procedura di apprendimento
% di un perceptron monostrato, che in questo caso è una porta logica OR
%
% LARA VIGNOTTO, mat 111794
% 13/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Carica i dati della rete già addestrata
load('OR_Trained_Network.mat')
%
N = 4;
for k = 1:N
    transposed_input = input(k, :)';
    weighted_sum = OR_Weight * transposed_input + OR_bias;
    output = Sigmoid(weighted_sum)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%