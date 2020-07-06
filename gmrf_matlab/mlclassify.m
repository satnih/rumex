function out = mlclassify(a, b, c, d)
	[rows, cols] = size(a);
	out = zeros(rows, cols);
	for i = 1:rows
		for j = 1:cols
			idx = find(min([a(i,j), b(i,j), c(i,j), d(i, j)]));
			out(i, j) = idx;
		end
	end
% 	out = label2rgb(out);
end