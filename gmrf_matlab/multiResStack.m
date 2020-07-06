function I_stack = multiResStack(I, filt_size, sigma)
	error(nargchk(3, 3, nargin))
	validateattributes(I, {'numeric'}, {'nonempty'})
	validateattributes(filt_size, {'numeric'}, {'nonempty', 'vector'})
	validateattributes(sigma, {'numeric'}, {'nonempty', 'vector'})
	
	[dummy, Ns] = size(sigma);
	[dummy, Nf] = size(filt_size);
	if(Ns ~= Nf)
		error('arguments fitl_sz and sigma should be of same length');
	end
	N = Ns;		
	I_stack = cell(N, 1);
	for i = 1:N
		H = fspecial('gaussian', filt_size(:, i), sigma(:,i));
		I_stack{i} = imfilter(I, H, 'replicate');
	end		
end
