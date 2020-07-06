function [idx] = strFindCell(STR, pat)
% STRFINDCELL returns the indices of the cells with strings having the
% pattern 'pat'
% STR is a cell array of strings

    idx1 = strfind(STR, pat);
    n = length(idx1);
    idx = zeros(0);
    for i = 1:n
        if isempty(idx1{i})
        else    
            idx = [idx;i];
        end
    end
end

