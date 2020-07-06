function out = downsample(A)
[nrow ncol] = size(A);
m = floor(nrow/4);
n = floor(ncol/4);
out = zeros(m, n);
for i = 1:m
    for j = 1:n
        out(i,j) = sum(sum(A(((i-1)*4 + 1):(i*4), ((j-1)*4 + 1):(j*4))));
    end
end
