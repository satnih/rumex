function [img, img_ref] = checkDim(img, img_ref)
%CHECKDIM: 
% Ensure that the dimensions of the refernce and the test image are 
% equal
    if(nargin < 2)
        error('???checkDim: incorrect number of arguments')
    end
    if(~(ismatrix(img) && ismatrix(img_ref)))
        error('???checkDim: functions expects both arguments to be matrix')
    end
    szref = size(img_ref);
    szimg = size(img);
    nrow = min(szref(1,1), szimg(1,1));
    ncol = min(szref(1,2), szimg(1,2));
    img = img(1:nrow, 1:ncol);
    img_ref = img_ref(1:nrow, 1:ncol);
end

