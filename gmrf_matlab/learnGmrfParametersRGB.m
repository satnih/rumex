%*************************************************************************
% Function to learn GMRF parameters of cla 'cla' by means of LS. The
% training data is determined based on the 'cla' value
% selected and used 
% to learn the parameters
% @param:
%	1) cla: cla of the parameters to be estimated
% @return:
%	A vector 'frts' containing the GMRF features beta(s) and nu
%*************************************************************************
function [gmrf_params_red, gmrf_params_green, gmrf_params_blue] = learnGmrfParametersRGB(cla, nhood_type)
    switch cla
        case 'rumex'
            train_data_path = '/u/21/hiremas1/unix/gmrf/data/aerial/train/rumex/*.png';
        case 'notrumex'
            train_data_path = '/u/21/hiremas1/unix/gmrf/data/aerial/train/notrumex/*.png';
         otherwise
            error('???learnGmrfParameters: fof file name error');
    end
	
	[flist, nfiles] = myGetFlist(train_data_path);
	% read images and stack them in a cell
	IMAGES_red = cell(nfiles, 1);
    IMAGES_green = cell(nfiles, 1);
    IMAGES_blue = cell(nfiles, 1);
    for iter = 1:nfiles
        a = double(imread(flist{iter}));        
        IMAGES_red{iter, 1} = a(:, :, 1);
        IMAGES_green{iter, 1} = a(:, :, 2);
        IMAGES_blue{iter, 1} = a(:, :, 3);
        
    end
	gmrf_params_red = gmrfFeaturesLseBlock(IMAGES_red, nhood_type);	
    gmrf_params_green = gmrfFeaturesLseBlock(IMAGES_green, nhood_type);	
    gmrf_params_blue = gmrfFeaturesLseBlock(IMAGES_blue, nhood_type);	
end
