function [ uxy ] = gibbsPosterior(y, x, nhood_type, lambda, param_r, param_g)
%GIBBSPOTENITAL: 
% Compute the Gibbs potential of y givn the labelling x with parameters
% nhood_type, lambda and the class parameters param_r and param_g
    
    [y, x] = checkDim(y, x);
    [rows cols] = size(y);
    ux = priorPotential(x, lambda);
    uyx = likelihoodPotential(y, x, param_r, param_g, nhood_type);
    uxy = (ux + uyx)/(rows * cols);
end

