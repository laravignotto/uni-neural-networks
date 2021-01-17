%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function processLabesFashionMNIST.m
% Estrae i dati dai file IDX in array MATLAB.
%
% VIGNOTTO LARA, mat 111794
% 27/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Y = processLabelsFashionMNIST(filename)
%   Apre in lettura il file IDX di etichette
    [fileID,errmsg] = fopen(filename,'r','b');
%   Controlla se il file può essere aperto correttament
    if fileID < 0   % fopen non è riuscito a aprire il file
        error(errmsg);
    end

%   Legge i dati da file binario e ottiene il magic number 
%   leggendo i primi 4 bytes.
    magicNum = fread(fileID,1,'int32',0,'b');
    if magicNum == 2049 % Magic number per le etichette
        fprintf('\nRead fashion MNIST label data...\n')
    end

    numItems = fread(fileID,1,'int32',0,'b');   % Numero di labels
    fprintf('Number of labels in the dataset: %6d ...\n',numItems);

%   Array contenente i valori 0-9 delle etichette
    Y = fread(fileID,inf,'unsigned char');

%     Y = categorical(Y);

    fclose(fileID); % Chiude il file
end