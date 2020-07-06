function [outim] = bwlargestblob(im,connectivity)
%
% 'bwlargestblob' reads in a 2-d binary image and outputs a binary image, retaining only the largest blob.
% 
% Usage: [outim] = bwlargestblob(im,connectivity)
% 
% im - 2-d binary image 
% conenctivity - Accepts 4/8 connectivity
% outim - Output binary image (with 1s and 0s)
% 
% Example:
% 
% im = imread('text.png');
% outim = bwlargestblob(im,8);
% figure; 
% subplot(1,2,1); imshow(im);
% subplot(1,2,2); imshow(255*outim);

if size(im,3)>1,
    error('bwlargestblob accepts only 2 dimensional images');
end

[imlabel totalLabels] = bwlabel(im,connectivity);
sizeblob = zeros(1,totalLabels);
for i=1:totalLabels,
    sizeblob(i) = length(find(imlabel==i));
end
largestBlobNo = [];
if(sum(sizeblob) ~= 0)
    [maxno largestBlobNo] = max(sizeblob);
end

outim = zeros(size(im),'uint8');
if isempty(largestBlobNo)
else
    outim(find(imlabel==largestBlobNo)) = 1;
end
    