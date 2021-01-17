%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funzione di attivazione SoftMax
%
% LARA VIGNOTTO, mat 111794
% 03/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function y = Softmax(x)
    ex = exp(x);
    y = ex/sum(ex);
end
