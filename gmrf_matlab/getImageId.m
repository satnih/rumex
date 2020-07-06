function [imgid] = getImageId(f, dlm1, dlm2)
    idx1 = strfind(f, dlm1);
    idx2 = strfind(f, dlm2);
    imgid = f((idx1(end)+1):(idx2-1));
end