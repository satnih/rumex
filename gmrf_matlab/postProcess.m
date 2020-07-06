function out = postProcess(a)
    a = bwareaopen(a , 30);
    c = logical(maxf(a, 25,'elliptic'));
    d = logical(medif(double(c), 30, 'elliptic'));   
    e = bwareaopen(d , 250);
    out = bwlargestblob(e,8);
end