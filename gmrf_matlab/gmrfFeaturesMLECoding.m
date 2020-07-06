function [ftrs] = gmrfFeaturesMLECoding(D, nhood_type)
	mu = mean(D(:));
	
	% zero mean image
	D = (D - mu);

	%isotropic, 1st order neighbourhood system
	Q = computeQmatrix(D, nhood_type); 
	Q = Q(:,:,1:(end-1));

	if(nhood_type == 1 || nhood_type == 2)
		Nb = 1;
	elseif(nhood_type == 3)
		Nb = 2;
	else
		Nb = 4;
	end
	disp(['Nb = ', num2str(Nb)]);

	Mn = size(Q(:,:,1), 1);
	Nn = size(Q(:,:,1), 2);
	
	Q1 = Q(:, :, 1);
	data = Q(:, :, Nb);
	clear Q;
	
	% split pixels according to coding method
	I = round(checkerboard(1, Mn/2, Nn/2)); % create a checkerboard matrix with the required coding scheme
	idx_code0 = find(I == 0); % indices of pixels belonging to code 0
	idx_code1 = find(I == 1); % indices of pixels belonging to code 1
	
	data_code0 = data(idx_code0); % pixels with code 0
	q_code0 = Q1(idx_code1); % its 1-order neighbours having code 1
	
	data_code1 = data(idx_code1); % pixels with code 1
	q_code1 = Q1(idx_code0); % its 1-order neighbours having code 0
	
	% ML estimate of a multivariate gaussian
	[beta_code0, nu_code0, resid_code0, info_code0] = mvnrmle(data_code0, q_code0);
	[beta_code1, nu_code1, resid_code1, info_code1] = mvnrmle(data_code1, q_code1);
	
	% average the estimates across different code
	beta = (beta_code0 + beta_code1)./2;
	nu = (nu_code0 + nu_code1)./2;
	ftrs = [beta', nu];
end
