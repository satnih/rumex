function N = getNoOfNbrs(nhood_type)
    % NEIGHBOURHOOD TYPE
    %----------------------------
    % 1 - 1st order isotropic 
    % 2 - 2nd order isotropic
    % 3 - 1st order anisotropic
    % 4 - 2nd order anisotropic
    %----------------------------
    if(nhood_type == 1 || nhood_type == 2)
        N = 1;
    elseif(nhood_type == 3)
        N = 2;
    elseif (nhood_type == 4)
        N = 4;
    end
end