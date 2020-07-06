function [energy] = gibbsPosteriorBatch(Y, X, EXP_PARAMETERS, nfile, nexp)
energy = zeros(nfile, nexp);
for j = 1:nexp    
    %-- extract parameters    
    lambda = EXP_PARAMETERS(j, 2);
    nhood_type = EXP_PARAMETERS(j, 3);
    sz = EXP_PARAMETERS(j, 4);
    sd = EXP_PARAMETERS(j, 5);
    beta_r = EXP_PARAMETERS(j, 6);
    nu_r = EXP_PARAMETERS(j, 7);
    mu_r = EXP_PARAMETERS(j, 8);
    beta_g = EXP_PARAMETERS(j, 9);
    nu_g = EXP_PARAMETERS(j, 10);
    mu_g = EXP_PARAMETERS(j, 11);
    
    param_r = [beta_r, nu_r, mu_r];
    param_g = [beta_g, nu_g, mu_g];
    
    h = fspecial('gaussian', sz, sd);        
    for i = 1:nfile
        % get largest blob of reference image
        x = X{i, 1};
        x = bwlargestblob(x, 8);

        % get image
        y = Y{i, 1};
        y =  imfilter(y, h, 'same', 'circular');  
        energy(i, j) = gibbsPosterior(y, x, nhood_type, lambda, param_r, param_g );
    end
end