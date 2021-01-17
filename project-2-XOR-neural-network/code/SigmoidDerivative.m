%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funzione SigmoidDerivative.m
%
% LARA VIGNOTTO, mat 111794
% 20/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = SigmoidDerivative(x)
    if (isscalar(x))
        y = x * (1-x);
    else
        y = x .* (1-x);
    end
end
