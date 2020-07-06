function [aout] = zeroCorrect(a)
if length(size(a)) < 2 || length(size(a)) > 3
    error('zeroCorrect: input should be a 2 or 3 dim array')
elseif length(size(a)) == 2    
    c = cellfun(@f, a, 'uniformoutput', 0);
    aout = c{1};
elseif length(size(a)) == 3
    c = num2cell(a, [1 2]);
    c = cellfun(@f, c, 'uniformoutput', 0);
    aout = cat(3, c{1},c{2},c{3});
end
end
function y = f(x)
    y = x;
    y(y==0) = mean(y(:));
end
