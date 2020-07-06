%*************************************************************************
% Function to compute the data costs of an image given the class 
% parameters
% @param:
%	1) D: input image
%	2) beta, nu: GMRF parameters
%	3) nhood_type: Neighbourhood System (see file computeQmatrix
%					for possible neighbourhood types)
% @return:
%	data cost matrix 'dc' of the image
%*************************************************************************
function [dc, f] = computeDc(D, beta, nu, mu,  nhood_type)
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
    D = double(D);
  	D = (D - mu);
	Q = computeQmatrix(D, nhood_type);
	Mn = size(Q(:, :, 1), 1);
	Nn = size(Q(:, :, 1), 2);
	dc = zeros(Mn, Nn, 1);
	f = zeros(Mn, Nn, 1);
	mu = zeros(Mn, Nn, 1);
% 	for i = 1:Mn
% 		for j = 1:Nn
% 			y_ij = Q(i, j, Nb + 1);        
% 			y_nij = reshape(Q(i, j, 1:(end-1)), 1, Nb);
% 			if nu ~= 0
% 				dc(i,j) = 0.5*(y_ij - beta*y_nij')^2/nu + 0.5*log(2*pi*nu);
% 			else
% 				error('ERROR in computeDc: division by zero');
% 			end
%  			f(i, j) = beta*(y_nij');
% 		end
%    end
    % vectorized version of the for loop above
    %     Q =   _ _     MnxNnx(Nb+1)
    %          |   |
    %         _|_  |
    %        |   |_|
    %        |   |/   
    %        |_ _|
    %           
    %   y_ij =   Mn*Nn x 1; pixel intensities
    %   y_nij =  Mn*Nn x Nb; neighbours of y_ij 
    %   r1 = 1 x Mn*Nn => beta*t(y_nij) => #,#,# * 1 ######... (matrix form)
    %                                      1,2,3   2 ######... 
    %                                              3 ######... 
    y_ij = reshape(Q(:,:,Nb+1),Mn*Nn,1);
    y_nij=reshape(Q(:,:,(1:Nb)),Mn*Nn,Nb);
    r1 = beta*y_nij';
    if(nu ~= 0)
        dc = 0.5*(y_ij - r1').^2/nu + repmat(0.5*log(2*pi*nu),Mn*Nn,1);
        dc = reshape(dc,Mn,Nn);
    else
        error('computeDc: divide by 0');
    end
    f = 1;
end
