function [ energy ] = gibbsLikelihood(y, x, param_r, param_g, nhood_type)
    if(nargin ~= 5)
        error('???likelihoodPotential: function expects 5 arguments')
    end
    %LIKELIHOODENERGY
    % compute the likelihood energy of the estimated segmentation xstar

    Nb = getNoOfNbrs(nhood_type);
    beta_r = param_r(1:(end-2));
    nu_r = param_r(end-1);
    mu_r = param_r(end);

    beta_g = param_g(1:(end-2));
    nu_g = param_g(end-1);
    mu_g = param_g(end);

    % compute Q matrix of both y and x
    Q = computeQmatrix(y, nhood_type);
    rows = size(Q, 1);
    cols = size(Q, 2);

    q = zeros(rows*cols, Nb+1);
    for i = 1:Nb
        q(:, i) = reshape(Q(:, :, i), rows*cols, 1);
    end


    % find indices of rumex and grass pixels
    x = x(1:rows, 1:cols);
    ind_r = find(x == 1);
    ind_g = find(x == 0);


    % energy of rumex pixels
    if(nhood_type == 1)
        mumat_r = ones(length(ind_r), Nb)*4*mu_r;
    elseif(nhood_type == 2)
        mumat_r = ones(length(ind_r), Nb)*8*mu_r;
    elseif(nhood_type == 3)
        mumat_r = ones(length(ind_r), Nb)*2*mu_r;
    elseif(nhood_type == 4)
        mumat_r = ones(length(ind_r), Nb)*2*mu_r;
    end
    
    yij_r = q(ind_r, end) - mu_r;
    ynij_r = q(ind_r, 1:(end-1)) - mumat_r;
    unij_r = beta_r * ynij_r'; % energy of neighbourhood
    if(nu_r ~= 0)
        er = 0.5*(yij_r - unij_r').^2/nu_r + repmat(0.5 *log(2 * pi * nu_r), size(ind_r, 1), 1);        
    else
        error('likelihoodEnergy: nu_r = 0, divide by 0 error');
    end

    % energy of grass pixels
    if(nhood_type == 1)
        mumat_g = ones(length(ind_g), Nb)*4*mu_g;
    elseif(nhood_type == 2)
        mumat_g = ones(length(ind_g), Nb)*8*mu_g;
    elseif(nhood_type == 3)
        mumat_g = ones(length(ind_g), Nb)*2*mu_g;
    elseif(nhood_type == 4)
        mumat_g = ones(length(ind_g), Nb)*2*mu_g;
    end
    
    yij_g = q(ind_g, end) - mu_g;
    ynij_g = q(ind_g, 1:(end-1)) - mumat_g;
    unij_g = beta_g * ynij_g'; % energy of neighbourhood
    if(nu_r ~= 0)
        eg = 0.5*(yij_g - unij_g').^2/nu_g + repmat(0.5 *log(2 * pi * nu_g), size(ind_g, 1), 1);        
    else
        error('likelihoodEnergy: nu_h = 0, divide by 0 error');
    end

    energy = (sum(er) + sum(eg)) / (rows * cols);
end

