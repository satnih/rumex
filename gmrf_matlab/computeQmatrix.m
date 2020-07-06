function [Q] = computeQmatrix(D, type)
%*************************************************************************
% Test function to evaluate the construction of Q matrix based on the
% GMRF neighbourhood
% @parm:
%	1) D: input image of size rows x cols 
%	2) type: type of the neighbourhood (1-isotropic1, 2-isotropic2, 
%			 3-anisotropci1, 4-anisotropic2)
% @return:
%	Q matrix 
% Neighbourhood Types:
% switch nhood_type
% case 1: 
%				  beta   
%					|    
%					|   
%				 	|  
%			------(x,y)-----beta
%				 	|  
%					|   
%					|    
% case 2:
%			beta  beta   beta
%			   \	|    /
%			    \	|   /
%			     \	|  /
%			------(x,y)-----beta
%			        |  
%			      	|   
%			     	|    
%
% case 3:
%				  beta2   
%					|    
%					|   
%				 	|  
%			------(x,y)-----beta1
%				 	|  
%					|   
%					|    
%
% case 4:
%			beta4 beta3 beta2
%			   \	|    /
%				\	|   /
%				 \	|  /
%			------(x,y)-----beta1
%				 /	|  \
%				/	|   \
%			   /	|    \
%
% case 5:
%			beta       beta
%			   \	     /
%				\	    /
%				 \	   /
%				  (x,y)
%				 /	   \
%				/	    \
%			   /	     \
%*************************************************************************
	L = 1;
	Ld = round(L / sqrt(2));
	[M, N] = size(D);
	switch type
		case 1
			Nb = 1;
% 			Q = zeros(M, N, Nb + 1);
% 			for i = (1+L):(M-L)
% 				for j = (1+L):(N-L)
% 				   Q(i, j, 1) = D(i - L, j) + D(i + L, j) + ...
% 						D(i, j - L) + D(i, j + L);
% 				end
%             end
            % vectorized version of the above loop
			Q = zeros(M, N, Nb + 1);
            Q(2:(end-1),2:(end-1),1) = ...
                D(2:(end-1),1:(end-2)) + D(2:(end-1),3:end) + ... % east, west
                D(1:(end-2),2:(end-1)) + D(3:end,2:(end-1)); % north, south
            
		case 2
% 			Nb = 1;
% 			Q = zeros(M, N, Nb + 1);
% 			for i = (1+L):(M-L)
% 				for j = (1+L):(N-L)
% 				   Q(i, j, 1) = D(i - L, j) + D(i + L, j) + ...
% 						D(i, j - L) + D(i, j + L) + ...
% 						D(i - Ld, j - Ld) + D(i + Ld, j + Ld) + ...
% 						D(i - Ld, j + Ld) + D(i + Ld, j - Ld);
% 				end				
% 			end
            % vectorized version --------------------------
			Nb = 1;
			Q = zeros(M, N, Nb + 1);
            Q(2:(end-1),2:(end-1),1) = ...
                D(2:(end-1),1:(end-2)) + D(2:(end-1),3:end) + ...
                D(1:(end-2),2:(end-1)) + D(3:end,2:(end-1)) + ...
                D(1:(end-2),1:(end-2)) + D(3:end,3:end) + ...
                D(1:(end-2),3:end) + D(3:end,1:(end-2));
%--------------neighbourhood range-----------------------------------------            
%           % COPY OF ABOVE CODE
%           len = 1;
% 			Nb = 1;
% 			Q = zeros(M, N, Nb + 1);
%             Q((len + 1): (end - len), (len + 1):(end - len),1)= ...
%                 D((len + 1):(end - len), 1:(end - 2*len)) + ...% west
%                 D((len + 1):(end - len), (2*len + 1):end) + ...% east                    
%                 D(1:(end - 2*len), (len + 1):(end - len)) + ...% north
%                 D((2*len + 1):end, (len + 1):(end - len)) +... % south
%                 D(1:(end - 2*len), 1:(end - 2*len)) + ...% north west
%                 D((2*len + 1):end, (2*len + 1):end) + ...% south east
%                 D(1:(end - 2*len), (2*len + 1):end) + ...% north east
%                 D((2*len + 1):end, 1:(end - 2*len));     % south west
%--------------neighbourhood range-----------------------------------------
        case 3
			Nb = 2;
			Q = zeros(M, N, Nb + 1);
			for i = (1+L):(M-L)
				for j = (1+L):(N-L)
				   Q(i, j, 1) = D(i - L, j) + D(i + L, j);
				   Q(i, j, 2) = D(i, j - L) + D(i, j + L);
				end
			end
		case 4
			Nb = 4;
			Q = zeros(M, N, Nb + 1);
			for i = (1+L):(M-L)
				for j = (1+L):(N-L)
				   Q(i, j, 1) = D(i - L, j) + D(i + L, j);
				   Q(i, j, 2) = D(i, j - L) + D(i, j + L);
				   Q(i, j, 3) = D(i - Ld, j - Ld) + D(i + Ld, j + Ld);
				   Q(i, j, 4) = D(i - Ld, j + Ld) + D(i + Ld, j - Ld);
				end
            end
		case 5
            % vectorized and specific version of the above loop
			Nb = 1;
			Q = zeros(M, N, Nb + 1);
            Q(2:(end-1),2:(end-1),1) = ...
                D(1:(end-2),1:(end-2)) + D(3:end,3:end) + ...
                D(1:(end-2),3:end) + D(3:end,1:(end-2)); 
		otherwise
			error('type is neighbourhood not mentioned');
	end
	Q(:,:,end) = D;
 	Q = Q((1+L):(M-L),(1+L):(N-L), :);
end
