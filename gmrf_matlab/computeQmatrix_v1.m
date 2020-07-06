function [Q] = computeQmatrix_v1(I, type, npoints, radius)
%*************************************************************************
% Test function to evaluate the construction of Q matrix based on the
% GMRF neighbourhood
% @parm:
%	1) I: input image of size rows x cols 
%	2) type: type of the neighbourhood (1-isotropic1, 2-isotropic2, 
%			 3-anisotropci1, 4-anisotropic2)
%   3) radius: distance from the central pixel
% @return:
%	Q matrix 
% Neighbourhood Types:
% case 1:
%			beta  beta   beta
%			   \	|    /
%			    \	|   /
%			     \	|  /
%			------(x,y)-----beta
%			        |  
%			      	|   
%			     	|    
%
% case 2:
%			beta4 beta3 beta2
%			   \	|    /
%				\	|   /
%				 \	|  /
%			------(x,y)-----beta1
%				 /	|  \
%				/	|   \
%			   /	|    \
%*************************************************************************
	[nrows, ncols, nchannels] = size(I);
    if nchannels > 1
        print('Converting RGB image to grey');
        I = rgb2gray(I);
    end
    Q = zeros(M, N, npoints + 1, nchannels);
	Q(:,:,end) = I;
	switch type
		case 1
            % roi coordinates 
            x = [radius, radius, ncols - radius, ncols - radius];
            y = [radius, nrows - radius, nrows - radius, radius];            
            
            mask = poly2mask(x, y, nrows, ncols); % roi mask 
            [cy, cx] = find(mask); % (x,y) coordinates of all pixels in roi mask
            ncentres = size(cx, 1);
            Q = zeros(nrows, ncols);
            for i = 1:ncentres
                Q(cy(i), cx(i), :) = getNeighbours(I, cx(i), cy(i), radius, npoints);                
            end    
		case 2
			for i = (1+L):(nrows-L)
				for j = (1+L):(ncols-L)
				   Q(i, j, 1) = I(i - L, j) + I(i + L, j);
				   Q(i, j, 2) = I(i, j - L) + I(i, j + L);
				   Q(i, j, 3) = I(i - Ld, j - Ld) + I(i + Ld, j + Ld);
				   Q(i, j, 4) = I(i - Ld, j + Ld) + I(i + Ld, j - Ld);
				end
            end
		otherwise
			error('type is neighbourhood not mentioned');
    end
 	Q = Q(cy(1):cy(end) , cx(1):cx(end), :);
end

function neighbours = getNeighbours(I, cx, cy, radius, npoints)
    [nrows, ncols, nchannels] = size(I);
    neighbours = zeros(npoints, nchannels);
    % npoints on the circle
    t = linspace(0, 2*pi, npoints);    
    % cartesian coordinates
    x = radius*cos(t)+cx;
    y = radius*sin(t)+cy;

    %interpolate to get pixel values at the location
    for i = 1:nchannels
        neighbours(:, i) = uint8(interp2(1:nrows,1:ncols, double(I(:,:,i)),x,y)) ; % red channel
    end
end
