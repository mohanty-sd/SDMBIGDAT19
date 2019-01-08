function xVec = r2sv(rVec,params)
%Convert real coordinates to standardized ones.
%X = R2SV(R,P)
%Takes standardized coordinates in X (coordinates of one point per row) and
%returns real coordinates in R using the range limits defined in P.rmin and
%P.rmax. The range limits can be different for different dimensions. (If
%they are same for all dimensions, use S2RS instead.)

%Soumya D. Mohanty
%May 2016
[nrows,ncols] = size(rVec);
xVec = zeros(nrows,ncols);
rmin = params.rmin;
rmax = params.rmax;
rngVec = rmax-rmin;
for lp = 1:nrows
    xVec(lp,:) = (rVec(lp,:)-rmin)./rngVec;
end