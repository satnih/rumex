function [imout] = preProcess(imin, varargin)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here
    iter = varargin{1}(1);
    cond = varargin{1}(2);
    lam = varargin{1}(3);
    opt = varargin{1}(4);
    
    a = double(adapthisteq(uint8(imin)));
    imout = anisodiff(a, iter, cond, lam, opt);
end

