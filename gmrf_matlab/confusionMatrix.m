function cm = confusionMatrix(a, ref)
	if(~islogical(a) || ~islogical(ref))
		error('Need logical arrays to compute confusion matrix');		
	end
	nclass = 2;
	cm = zeros(nclass, nclass);
	for i = 1:nclass
		for j = 1:nclass
			cm(i,j) = sum((a(:) == (i-1) & ref(:) == (j-1)));
		end
	end
end

