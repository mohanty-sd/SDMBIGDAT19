%% Test harness for CRCBPSO 
% The fitness function called is CRCBPSOTESTFUNC. 
ffparams = struct('rmin',-100,...
                     'rmax',100 ...
                  );
% Fitness function handle.
fitFuncHandle = @(x) crcbpsotestfunc(x,ffparams);
%%
% Call PSO.
rng('default')
psoOut = crcbpso(fitFuncHandle,2);

%% Estimated parameters
% Best standardized and real coordinates found.
stdCoord = psoOut.bestLocation;
[~,realCoord] = fitFuncHandle(stdCoord);
disp(['Best location:',num2str(realCoord)]);
disp(['Best fitness:', num2str(psoOut.bestFitness)]);

%%
% Call PSO with optional inputs
rng('default')
psoOut = crcbpso(fitFuncHandle,2,[],2);
%Plot the trajectory of the best particle
figure;
plot(psoOut.allBestLoc(:,1),psoOut.allBestLoc(:,2),'.-'));
figure;
plot(psoOut.allBestFit);
