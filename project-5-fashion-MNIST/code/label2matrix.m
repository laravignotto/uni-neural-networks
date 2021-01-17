%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funzione label2matrix
% Conversione di un insieme di etichette in forma matriciale.
% label: insieme di etichette da convertire
%   
% LARA VIGNOTTO, mat 111794
% 07/12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function matrix = label2matrix(label)
%
%   Conversione di formato di un'etichetta per avere         
%   categorie numerate da 1 a 10 e non da 0 a 9
	label(label == 0) = 10;
%   Inizializzazione delle matrici
    matrix = zeros(length(label),10);
%
    for i = 1:length(label)
        matrix(i, label(i)) = 1;
    end
%
    end  %  fine della function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%