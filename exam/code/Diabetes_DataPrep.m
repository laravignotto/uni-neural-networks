%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function Diabetes_DataPrep.m
% Accesso al dataset diabetes.csv;
% Definizione dataset locale (training set e correct output).
%
% VIGNOTTO LARA, mat 111794
% 15/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [TrainingData, TrainingLabels, ValidationData, ValidationLabels] = ...
    Diabetes_DataPrep()
%
    DataSet = readmatrix('diabetes.csv');
%
%%%%%%%%%%%%%%%%%%% Splitting
%   Numero totale dei campioni
    numtotData = 768;
%
%	Percentuale di splitting
	training_perc = 0.8; 
%
%	Randomizzazione degli indici delle immagini
	split_perm = randperm(numtotData);
%
%	Cardinalità dell'insieme di apprendimento (training)
	training_cardin = floor(numtotData * training_perc);  
%
%	Definizione degli insiemi di apprendimento 
%	e delle relative etichette, tutti randomizzati;
    TrainingData = DataSet(split_perm(1:training_cardin),1:8);
    TrainingLabels = DataSet(split_perm(1:training_cardin),9);
%
%	Definizione degli insiemi di collaudo 
%	e delle relative etichette, tutti randomizzati
    ValidationData = DataSet(split_perm(training_cardin+1:end),1:8);
    ValidationLabels = DataSet(split_perm(training_cardin+1:end),9);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%