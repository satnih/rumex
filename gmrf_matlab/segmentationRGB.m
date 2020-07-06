clear
addpath('/u/21/hiremas1/unix/gmrf/src/GCmex2.0/')

%{
% Optimal parameters from paper
lambda = 1.8;
nhood_type = 3;
h = fspecial('gaussian', 4, 0.7);
beta_r = [0.27256	0.24199];
nu_r = 11.115;
mu_r = 148.41;
beta_g = [0.2828, 0.26583];
nu_g = 47.223;
mu_g = 99.289;
%}

%% parameters learnt from ariel image with RGB Images

% nhood_type = 3;
% h = fspecial('gaussian', 5, 0.2);
% beta_r = [0.2951, 0.2480];
% nu_r = 65.0267;
% mu_r = 147.2767;
% 
% beta_g = [0.2822, 0.2378];
% nu_g = 14.9666;
% mu_g = 103.6365;

nhood_type = 2;
h = fspecial('gaussian', 5, 0.2);

% % learnt from train folder rumex
% beta_r = 0.1389;
% nu_r = 87.2604;
% mu_r = 147.2767;

%{
% learnt from train folder rumex01
beta_red_r = 0.1385;  
nu_red_r = 106.3066; 
mu_red_r = 110.1826;

beta_green_r = 0.1399;
nu_green_r = 104.5279; 
mu_green_r = 132.0498;

beta_blue_r = 0.1412;  
nu_blue_r = 91.4543; 
mu_blue_r = 40.5929;
%}

lambda = 5;
% learnt from train folder rumex
beta_red_r = 0.1389;  
nu_red_r = 87.2628; 
mu_red_r = 147.2767;

beta_green_r = 0.1402;
nu_green_r = 75.2244; 
mu_green_r = 162.9523;

beta_blue_r = 0.1387;  
nu_blue_r = 76.5910; 
mu_blue_r = 66.5404;
%}

% grass parameters
beta_red_g = 0.1299;
nu_red_g = 13.7815;
mu_red_g = 110.3000;

beta_green_g = 0.1326;
nu_green_g = 11.8819;
mu_green_g = 116.7438;

beta_blue_g = 0.1332;
nu_blue_g = 14.0470;
mu_blue_g = 39.36;


%% load image and compute data cost
im = imread('/u/21/hiremas1/unix/gmrf/data/aerial/train/field_2_rumex_ortho_15m_4_se.png');

%{
x_crop_start = 300;
x_crop_end = size(im, 2);

y_crop_start = 1;
y_crop_end = 2500;
%}

x_crop_start = 1;
x_crop_end = 2800;

y_crop_start = 1;
y_crop_end = 1700;

im = im(y_crop_start:y_crop_end, x_crop_start:x_crop_end, :);
a = double(im);
red = a(:, :, 1);
green = a(:, :, 2);
blue = a(:, :, 3);


[dc_red_r, dummy] = computeDc(red, beta_red_r, nu_red_r, mu_red_r, nhood_type);
[dc_green_r, dummy] = computeDc(green, beta_green_r, nu_green_r, mu_green_r, nhood_type);
[dc_blue_r, dummy] = computeDc(blue, beta_blue_r, nu_blue_r, mu_blue_r, nhood_type);

[dc_red_g, dummy] = computeDc(red, beta_red_g, nu_red_g, mu_red_g, nhood_type);
[dc_green_g, dummy] = computeDc(green, beta_green_g, nu_green_g, mu_green_g, nhood_type);
[dc_blue_g, dummy] = computeDc(blue, beta_blue_g, nu_blue_g, mu_blue_g, nhood_type);

dc_r = dc_green_r + dc_red_r + dc_blue_r;
dc_g = dc_green_g + dc_red_g + dc_blue_g;

%   construct input file to graph cut
[nrows,ncols] = size(dc_r);
Dc = zeros(nrows, ncols, 2);
Dc(:, :, 1) = dc_r;
Dc(:, :, 2) = dc_g;
clear('dc1', 'dc2')

nclasses = 2;
Sc = ones(nclasses) - eye(nclasses);
% smoothness cost matrices
%gch = GraphCut('open', Dc, 10*Sc, exp(-Vc*5), exp(-Hc*5));
gch = GraphCut('open', Dc, 5*Sc); % open a hadle to graphcut objct
[gch L] = GraphCut('expand',gch); % perform segmentation
gch = GraphCut('close', gch); % close handle
imshow(logical(L))

%{
% show results
imshow(im);
hold on;
PlotLabels(L);
%-----------------------------------------------%
function ih = PlotLabels(L)

    L = single(L);

    bL = imdilate( abs( imfilter(L, fspecial('log'), 'symmetric') ) > 0.1, strel('disk', 1));
    LL = zeros(size(L),class(L));
    LL(bL) = L(bL);
    Am = zeros(size(L));
    Am(bL) = .5;
    ih = imagesc(LL); 
    set(ih, 'AlphaData', Am);
    colorbar;
    colormap 'white';
end
%}
%{
%% Compute Rumex Centre
[flist, nfile] = myGetFlist(fofresults_gc);
CENTRE = zeros(nfile,2);
AREA = zeros(nfile, 1);
POSTPROCESSED_IMGS= cell(nfile,1);
for i = 1:nfile
    i
    f = flist{i};
    im = logical(transpose(1-imread(f)));
    if(sum(im(:)) == size(im,1)*size(im,2) || sum(im(:)) == 0 )
        disp(['segmentation failed ', num2str(i)])
        continue;
    end    
    im = logical(postProcess(im));    
    if(sum(im(:)) == size(im,1)*size(im,2) || sum(im(:)) == 0 )
        disp(['segmentation failed ', num2str(i)])
        continue;
    else
        POSTPROCESSED_IMGS{i,1} = im;
        prop = regionprops(im, 'Area', 'Centroid');
        CENTRE(i,:) = prop.Centroid;
        AREA(i, 1) = prop.Area;
    end    
end

save([store_resultsdir,'/',expname,'/',expname,'_','POSTPROCESSED_IMGS'], 'POSTPROCESSED_IMGS');
save([store_resultsdir,'/',expname,'/',expname,'_','CENTRE'], 'CENTRE');
save([store_resultsdir,'/',expname,'/',expname,'_','AREA'], 'AREA');

%% Distance Error
% load reference image centre
% clear
% load(['/home/santosh/phd/201001_rumex/data/testset1/group1/',lower(resln),'/reference/CENTRE_REF.mat'])
load('/home/santosh/phd/201001_rumex/gmrf/src/matlab_code/final_experiments/results/AREA_AND_CENTRE_REF_TESTSET1.mat')
load([store_resultsdir,'/',expname,'/',expname,'_','CENTRE']);

c = CENTRE;
cr = CENTRE_REF_TESTSET1;
dist = sqrt(sum(((cr - c).^2),2));
mu = mean(dist)'

%}