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
function [gmrf_parameters] = learnGmrfParameters(cla, nhood_type)
    switch cla
        case 'rumex'
            train_data_path = '/u/21/hiremas1/unix/gmrf/data/aerial/train/rumex01/*.png';
        case 'notrumex'
            train_data_path = '/u/21/hiremas1/unix/gmrf/data/aerial/train/notrumex/*.png';
         otherwise
            error('???learnGmrfParameters: fof file name error');
    end
	
	[flist, nfiles] = myGetFlist(train_data_path);
	% read images and stack them in a cell
	IMAGES = cell(nfiles, 1);
    for iter = 1:nfiles
        a = double(imread(flist{iter}));
        a = myIm2grey(a);        
        IMAGES{iter, 1} = a;
    end
	gmrf_parameters = gmrfFeaturesLseBlock(IMAGES, nhood_type);	
end
