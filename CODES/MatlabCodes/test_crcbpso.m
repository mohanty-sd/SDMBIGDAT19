%% Test harness for CRCBPSO 
% The fitness function called is CRCBPSOTESTFUNC. 
nDim = 20; %Dimensionality of the search space
rmin = -10;% lower bound of search space coordinate
rmax = 10; %Upper bound of search space coordinate
ffparams = struct('rmin',rmin,...
                     'rmax',rmax ...
                  );
% Fitness function handle.
fitFuncHandle = @(x) crcbpsotestfunc(x,ffparams);

%% Default PSO settings
disp('Default PSO settings');
disp(crcbpso());

%%
% Call PSO with default settings
disp('Calling PSO with default settings and no optional inputs')
rng('default')
tic;
psoOut1 = crcbpso(fitFuncHandle,nDim);
toc;
% Call PSO with default settings but return more information
disp('Calling PSO with default settings and optional inputs')
rng('default')
tic;
psoOut1 = crcbpso(fitFuncHandle,nDim,[],2);
toc;
% Best standardized and real coordinates found.
stdCoord = psoOut1.bestLocation;
[~,realCoord] = fitFuncHandle(stdCoord);
disp([' Best location:',num2str(realCoord)]);
disp([' Best fitness:', num2str(psoOut1.bestFitness)]);

%%
% Override default PSO parameters 
disp('Overriding default PSO parameters');
rng('default');
psoParams = struct();
psoParams.maxSteps = 30000; disp(['Changing maxSteps to:',num2str(psoParams.maxSteps)]);
psoParams.maxVelocity = 0.9; disp(['Changing maxVelocity to:',num2str(psoParams.maxVelocity)]);
tic;
psoOut2 = crcbpso(fitFuncHandle,nDim,psoParams,2);
toc;

%% Results
figure;
plot(psoOut1.allBestFit);
xlabel('Iteration number');
ylabel('Global best fitness');
title('Default PSO settings');
figure;
plot(psoOut2.allBestFit);
xlabel('Iteration number');
ylabel('Global best fitness');
title('Non-default PSO settings');
if nDim == 2
    %Plot the trajectory of the best particle
    figure;
    hold on;
    %Contour plot of the fitness function
    %=======================
    %X and Y grids
    xGrid = linspace(rmin,rmax,500);
    yGrid = linspace(rmin,rmax,500);
    [X,Y] = meshgrid(xGrid,yGrid);
    %Standardize
    Xstd = (X-rmin)/(rmax - rmin);
    Ystd = (Y-rmin)/(rmax - rmin);
    %Get fitness values
    fitVal4plot = fitFuncHandle([Xstd(:),Ystd(:)]);
    %Reshape array of fitness values
    fitVal4plot = reshape(fitVal4plot,size(X));
    contour((xGrid-rmin)/(rmax-rmin), (yGrid - rmin)/(rmax - rmin), fitVal4plot);
    %========================
    plot(psoOut2.allBestLoc(:,1),psoOut2.allBestLoc(:,2),'.-');
    title('Trajectory of the best particle');
    figure;
    title('Plot of fitness function');
    surf(X,Y,fitVal4plot); shading interp;
end
stdCoord = psoOut2.bestLocation;
[~,realCoord] = fitFuncHandle(stdCoord);
disp([' Best location:',num2str(realCoord)]);
disp([' Best fitness:', num2str(psoOut2.bestFitness)]);
