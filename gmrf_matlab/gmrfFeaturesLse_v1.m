function [ftrs] = gmrfFeaturesLse_v1(a, nhood_type)
	if(nhood_type == 1 || nhood_type == 2)
		Nb = 1;
	elseif(nhood_type == 3)
		Nb = 2;
	else
		Nb = 4;
	end
% 	disp(['Nb = ', num2str(Nb)]);
	a = a - mean(a(:));
	Q = computeQmatrix(a, Nb);
	a = Q(:, :, end);
	[M, N] = size(a);
	b = reshape(a, M*N, 1);
	A = zeros(M*N, Nb);
	for i = 1:Nb
		A(:, i) = reshape(Q(:, :, i), M*N, 1);
	end
	x = A\b;
	nu = sum((A*x - b).^2)/(M*N);
	ftrs = [x', nu];
% 	ftrs = nu;
end
