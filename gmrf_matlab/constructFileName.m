function out = constructFileName(id, expname, fmt)
    prefix = [fmt,'_'];
    postfix = ['_',id];
    ext = ['.',fmt];
    out = [prefix, expname, postfix, ext];    
end