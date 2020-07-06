function kappa = computeKappa(cm, nrow, ncol)
	s1 = 0;
	nclass = 2;
	for i = 1:nclass
		s1 = s1 + sum(cm(i,:))*sum(cm(:,i));
	end
	kappa = (nrow*ncol*sum(diag(cm)) - s1) / ((nrow*ncol)^2 - s1);
end