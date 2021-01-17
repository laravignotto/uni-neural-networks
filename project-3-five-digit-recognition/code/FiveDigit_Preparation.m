%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script FiveDigit_Preparation.m
%
% LARA VIGNOTTO, mat 111794
% 10/11/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [image] = FiveDigit_Preparation(image, ntest, dens)
    num_ripetizioni = (ntest/5);
    i=1;
    for j=1:num_ripetizioni
        for k=1:5 % numero immagini di input (1,2,3,4,5)
            image(:,:,i) = imnoise(image(:,:,k),'salt & pepper',dens);
            i = i+1;
        end
    end
%
end