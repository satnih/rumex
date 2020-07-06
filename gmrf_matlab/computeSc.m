%*************************************************************************
% Function to compute the smoothness costs of the image given the class 
% parameters
% @param:
%	1) D: input image
%	2) beta, nu: GMRF parameters
%	3) nhood_type: Neighbourhood System (see file computeQmatrix
%					for possible neighbourhood types)
% @return:
%	data cost matrix 'dc' of the image
%*************************************************************************
function [sch, scv] = computeSc(D, beta, nu, nhood_type)
	if nhood_type == 1 || nhood_type == 2 || nhood_type == 5
		Nb = 1;
	elseif nhood_type == 3
		Nb = 2;
	else
		Nb = 4;
	end
% 	disp(['nhood_type:', num2str(nhood_type),' Nb:', num2str(Nb)]);
  	mu = mean(D(:));
  	D = (D - mu);
	Q = computeQmatrix(D, nhood_type);
	Mn = size(Q(:, :, 1), 1);
	Nn = size(Q(:, :, 1), 2);
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
    %   y_nij =  Mn*Nn x Nb; neighbours of y_ij 
    %                                      1 2 3 
    %   r1 = 1 x Mn*Nn => beta*t(y_nij) => #,#,# x 1 ######...r (matrix form)
    %                                      cols    2 ######...o 
    %                                              3 ######...w
    %                                                         s              
    y_nij = reshape(Q(:,:,Nb+1),Mn*Nn,Nb);
    f = beta*y_nij';
    f = reshape(f, Mn, Nn);
    % assuming 1st order neighbourhood
    sch = abs(f(:,1:(end-1)) - f(:,2:end));
    scv = abs(f(1:(end-1),:) - f(2:end,:));
end
