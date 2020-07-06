function [] = dlcWrite(fname, data, delim )
%DLCWRITE Summary of this function goes here
%   Detailed explanation goes here
    if(nargin ~= 3)
        error('dlcwrite: arguments, fname, data and delm, expected')
    end
    
    if(~iscell(data))
        error('dlcwrite: data must be a cell array');
    end
    
    [nrow, ncol] = size(data);
    fid = fopen(fname,'w+');
    if isempty(fid)
        error('???dlcwrite: could not open file');
    end    
    for i = 1:nrow
        fprintf(fid,['%s',delim], data{i, 1:(end-1)});
        fprintf(fid,'%s\n',data{i, end});
    end
    fclose(fid);
end

