function [flist, nfiles] = myGetFlist(train_data_path)    
    flist = ls(train_data_path);
    flist = strsplit(flist, '\n');
    flist = flist(1:(end-1));
    nfiles = length(flist);    
end
