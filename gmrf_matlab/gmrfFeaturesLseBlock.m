%*************************************************************************
% Function to estimate the GMRF parameters from a set of images 
% @param:
%	1) IMAGES: class of the parameters to be estimated
%	2) nhood_type: Neighbourhood system (see file computeQmatrix
%					for possible neighbourhood types)
% @return:
%	A vector 'frts' containing the GMRF features beta(s) and nu
% 
%*************************************************************************
function [gmrf_parameters] = gmrfFeaturesLseBlock(IMAGES, nhood_type)
	L = 1;
	nimages = size(IMAGES, 1);
    
    % NEIGHBOURHOOD TYPE
    %----------------------------
    % 1 - 1st order isotropic 
    % 2 - 2nd order isotropic
    % 3 - 1st order anisotropic
    % 4 - 2nd order anisotropic
    %----------------------------
    if(nhood_type == 1 || nhood_type == 2)
		Nb = 1;
	elseif(nhood_type == 3)
		Nb = 2;
    elseif (nhood_type == 4)
		Nb = 4;
	end
% 	disp(['nhood_type:', num2str(nhood_type),' Nb:', num2str(Nb)]);

	C = zeros(Nb, Nb);
	v = zeros(Nb, 1);
	tmp = zeros(1,1);
	
	% compute global mean of all the images
	for k = 1:nimages
		img = IMAGES{k};
		[M, N] = size(img);
		tmp = cat(1, tmp, reshape(img, M*N,1));		
	end
	mu = mean(tmp);
	
	% construct the covariance matrix
	for k = 1:nimages		
		D = IMAGES{k};
		D = (D - mu);
        Q = computeQmatrix(D, nhood_type); 
		Q = Q(:,:,1:(end-1));

		Mn = size(Q(:,:,1), 1);
		Nn = size(Q(:,:,1), 2);

		for i = 1:Mn
			for j = 1:Nn			
				qi_dash = reshape(Q(i,j,:),1, Nb);
				qi = reshape(Q(i,j,:), Nb, 1);
				C = C + qi * qi_dash;			
				v = v + qi * D(i + L, j + L);
			end
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
	gmrf_parameters = [beta', nu, mu];
end
