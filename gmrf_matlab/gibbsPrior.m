function [ ux ] = gibbsPrior(x,lambda)
    %COMPUTESC smoothness cost of an image
    if(nargin ~= 2)
        error('???computeSc: function expcts two arguments')
    end
    if( ~ismatrix(x))
        error('???computeSc: argument must be a matrix')
    end
    
    [rows, cols] = size(x);
    
    E = circshift(x, [0, 1]);
    W = circshift(x, [0, -1]);
    N = circshift(x, [1, 0]);
    S = circshift(x, [-1, 0]);
    NE = circshift(x, [-1, 1]);
    SW = circshift(x, [1, -1]);
    NW = circshift(x, [-1, -1]);
    SE = circshift(x, [1, 1]);

    ch = xor(x, E) + xor(x, W);
    cv = xor(x, N) + xor(x, S);
    cd1 = xor(x, NE) + xor(x, SW);
    cd2 = xor(x, NW) + xor(x, SE);

    ux = (sum(sum(ch+cv+cd1+cd2))*lambda)/(rows * cols);
end

