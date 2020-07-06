function [ftrs] = gmrfFeaturesLse(D, nhood_type)
	L = 1;
	
	mu = mean(D(:));
	
	D = (D - mu);

	Q = computeQmatrix(D, nhood_type);
	Q = Q(:,:,1:(end-1));

	if(nhood_type == 1 || nhood_type == 2)
		Nb = 1;
	elseif(nhood_type == 3)
		Nb = 2;
	else
		Nb = 4;
	end
% 	disp(['Nb = ', num2str(Nb)]);

	Mn = size(Q(:,:,1), 1);
	Nn = size(Q(:,:,1), 2);
	
	C = zeros(Nb, Nb);
	v = zeros(Nb, 1);

	for i = 1:Mn
		for j = 1:Nn			
			qi_dash = reshape(Q(i,j,:),1, Nb);
			qi = reshape(Q(i,j,:), Nb, 1);
			C = C + qi * qi_dash;			
			v = v + qi * D(i + L, j + L);
		end
	end

	Npix = Mn*Nn;

	C = C/Npix;
	v = v/Npix;

	beta = inv(C) * v;
	temp = 0;
	for i = 1:Mn
		for j = 1:Nn
 			qi_dash = reshape(Q(i,j,:),1, Nb);
 			temp = temp + (D(i+L,j+L) - qi_dash*beta)^2;			
		end
	end

 	nu = temp/Npix;
%     ftrs = nu;
	ftrs = [beta', nu, mu];
end
