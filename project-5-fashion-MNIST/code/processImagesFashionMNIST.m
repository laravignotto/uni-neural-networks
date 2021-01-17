%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function processImagesFashionMNIST.m
% Estrae i dati dai file IDX in array MATLAB.
%
% VIGNOTTO LARA, mat 111794
% 27/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function X = processImagesFashionMNIST(filename)
%   Apre in lettura il file IDX di immagini 
    [fileID,errmsg] = fopen(filename,'r','b');
%   Controlla se il file può essere aperto correttamente
    if fileID < 0   % fopen non è riuscito a aprire il file
        error(errmsg);
    end

%   Legge i dati da file binario e ottiene il magic number 
%   leggendo i primi 4 bytes. 
    magicNum = fread(fileID,1,'int32',0,'b');
    if magicNum == 2051 % Magic number per le immagini
        fprintf('\nRead fashion MNIST image data...\n')
    end

%   Legge i successivi 3 set di 4 bytes, che ritornano 
%   il numero di immagini, righe e colonne
    numImages = fread(fileID,1,'int32',0,'b');  % Numero di immagini
    fprintf('Number of images in the dataset: %6d ...\n',numImages);
    numRows = fread(fileID,1,'int32',0,'b');    % Numero di righe
    numCols = fread(fileID,1,'int32',0,'b');    % Numero di colonne

%   Legge i dati di immagine
    X = fread(fileID,inf,'unsigned char');

%   Fa il reshape dell'array e scambia le prime due dimensioni
%   perché i dati sono stati letti per colonna
    X = reshape(X,numCols,numRows,numImages);
    X = permute(X,[2 1 3]);
%   Divide i valori dei pixel per 255 per normalizzare nel range [0,1] 
    X = X./255;
    X = reshape(X, [28,28,1,size(X,3)]); % Forma di matrice
    X = dlarray(X, 'SSCB'); % Converte l'array 3-D in un dlarray 4-D

    fclose(fileID); % Chiude il file
end