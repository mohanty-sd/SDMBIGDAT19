%% Test harness for CRCBPSO 
% The fitness function called is CRCBPSOTESTFUNC. 
ffparams = struct('rmin',-100,...
                     'rmax',100 ...
                  );
% Fitness function handle.
fitFuncHandle = @(x) crcbpsotestfunc(x,ffparams);
%%
% Call PSO.
rng('default');
psoOut = crcbpso(fitFuncHandle,2);

%% Estimated parameters
% Best standardized and real coordinates found.
stdCoord = psoOut.bestLocation;
[~,realCoord] = fitFuncHandle(stdCoord);
disp(realCoord);

%% Obtaining more information
% We keep the default PSO parameters, hence the third input argument is
% empty.
rng('default');
psoOut = crcbpso(fitFuncHandle,2,[],2);
figure;
plot(psoOut.allBestFit);
figure;
%Caution: The figure can only be made for a 2D search space
plot(psoOut.allBestLoc(:,1),psoOut.allBestLoc(:,2),'o-');