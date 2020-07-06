% RGB to monochrome
function [aout] = myIm2grey(a, varargin)
    if(nargin == 1)
        type = 1;
    elseif(nargin == 2)
        type = varargin{1};
    end
    if isempty(size(a, 3))
        error('myIm2grey: Function expects rgb image')
    end
    switch type
        case 1 % red
            aout = a(:,:,1);
        case 2 % green
            aout = a(:,:,2);
        case 3 % blue
            aout = a(:,:,3);
        case 4 % excess green
            aout = (2*a(:,: , 2) - (a(:,:,1) + a(:,:,3)));
        case 5 % grey
            aout = (a(:,:,1) + a(:,:,2) + a(:,:,3))/3;
    end
    h = fspecial('gaussian', 5, 0.2);
    aout = imfilter(aout, h, 'same', 'circular' ); 
end