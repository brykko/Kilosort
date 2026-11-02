cd('C:\code\github\neuropixels');
addpath('C:\code\github\neuropixels');
parallel.gpu.enableCUDAForwardCompatibility(true);
d = gpuDevice;
fprintf('GPU %s, compute %s, driver %.1f, toolkit %.1f\n', d.Name, d.ComputeCapability, d.DriverVersion, d.ToolkitVersion);
try
    x = gpuArray.rand(1000,'single'); y = x*x; fprintf('gpuArray mtimes ok: %g\n', gather(sum(y(:))));
catch e
    fprintf('gpuArray failed: %s\n', e.message);
end
SpikeSort.setKilosortPath(2.5);
which mexWtW2
nt0 = 61; Nfilt = 64; Nchan = 32;
W = gpuArray.randn(nt0, Nfilt, 3, 'single'); U = gpuArray.randn(Nchan, Nfilt, 3, 'single');
try
    WtW = getMeWtW(W, U);
    fprintf('mexWtW2 ok: size %s, sum %g\n', mat2str(size(WtW)), sum(WtW(:)));
    % CPU check of one lag-0 product
    Wc = gather(W); Uc = gather(U); ref = 0;
    for i=1:3, for j=1:3, ref = ref + (Uc(:,1,i)'*Uc(:,2,j)) * (Wc(:,1,i)'*Wc(:,2,j)); end, end
    fprintf('lag0 (1,2): mex %g, cpu %g\n', WtW(1,2,nt0), ref);
catch e
    fprintf('mexWtW2 failed: %s\n', e.message);
end
