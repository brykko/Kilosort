function ks25_sort(fbinary, metaFile, outDir, seed, testOps)
%KS25_SORT Kilosort 2.5 "standard" sort of one excerpt, as the neuropixels pipeline runs it.
%   KS25_SORT(FBINARY, METAFILE, OUTDIR, SEED) sorts the int16 binary FBINARY (385 channels: 384
%   data channels plus sync) and writes rezToPhy output to OUTDIR. METAFILE is the source
%   recording's SpikeGLX AP metadata, from which the channel map is built.
%
%   Mirrors SpikeSort.Kilosort.runKilosortClustering > runKilosortMatlab of the neuropixels repo
%   (C:\code\github\neuropixels): the same path setup (Shared.configureStageDependencies,
%   SpikeSort.setKilosortPath(2.5)), channel map (probes.ChannelMap from the SpikeGLX shank map,
%   as Datasets.DatasetSgl builds it; reference sites disconnected; no bad channels), ops preset
%   (kilosort_config_v2_5("standard")), steps (preprocessDataSub, datashift2(rez, 1),
%   learnAndSolve8b, autocurate) and export (rezToPhy). Differences, none of which change the sort
%   except the first:
%   - learnAndSolve8b's seed is SEED (the replicate index); the pipeline always passes 1, so
%     replicate 1 is the pipeline's sort. The seed sets the random order of the batches used to
%     learn templates;
%   - ops.fig = 0: no diagnostic figures (ops.fig only draws them);
%   - ops.curationTrackingTimeRange is not set: autocurate stores it in rez.ops and nothing reads it;
%   - fproc.dat is written to OUTDIR and deleted at the end; intermediate rez files are not saved.
%
%   KS25_SORT(..., TESTOPS) overrides ops fields with those of the struct TESTOPS (smoke tests only).
%
%   Run with MATLAB R2025a: matlab -batch "ks25_sort('<bin>', '<meta>', '<outdir>', 1)"
%   The RTX 5070 Ti (compute 12.0) needs CUDA forward compatibility in R2025a.

npxRoot = 'C:\code\github\neuropixels';
addpath(npxRoot);
Shared.configureStageDependencies('SpikeSort');
parallel.gpu.enableCUDAForwardCompatibility(true);
gpu = gpuDevice();
SpikeSort.setKilosortPath(2.5);

if ~isfolder(outDir), mkdir(outDir); end
log = Shared.initializeLogging(char(outDir), SpikeSort.Consts.LOG_FILE_NAME);
log('i', 'KS2.5 standard sort of "%s", seed %u, pipeline version %s, GPU %s (compute %s)', ...
    fbinary, seed, npxVersion(), gpu.Name, gpu.ComputeCapability);

% Channel map, as runKilosortClustering > createChanMap builds it
meta = fileread(metaFile);
tok = regexp(regexp(meta, '~snsShankMap=([^\s]*)', 'once', 'tokens'), ...
    '\((\d*):(\d*):(\d*):[0|1]\)', 'tokens');
shankMap = cellfun(@str2double, vertcat(tok{1}{:})) + 1;     % 1-based shank, col, row
nSavedChans = str2double(regexp(meta, 'nSavedChans=(\d*)', 'once', 'tokens'));
assert(size(shankMap, 1) == 384 && nSavedChans == 385, 'expected a 3A option 3 recording');
ielec = (shankMap(:, 3) - 1) * 2 + shankMap(:, 2);
chanMap = probes.helpers.createKilosortChannelMap( ...
    probes.ChannelMap(probes.models.Imec_Neuropixels_3a_opt3, ielec));
chanMapFile = fullfile(outDir, SpikeSort.Consts.CHANNELMAP_FILE_NAME);
save(chanMapFile, '-struct', 'chanMap', '-v7');

% ops, as runKilosortMatlab sets them
ops = SpikeSort.Kilosort.Config.kilosort_config_v2_5("standard");
ops.ksVersion = 2.5;
ops.prefiltered = false;
ops.debug = max(ops.debug, false);
ops.fig = 0;
ops.trange = [0, Inf];
ops.rootZ = char(outDir);
ops.NchanTOT = nSavedChans;
ops.chanMap = char(chanMapFile);
ops.fbinary = char(fbinary);
ops.fproc = fullfile(char(outDir), 'fproc.dat');
if nargin >= 5
    for f = string(fieldnames(testOps))'
        ops.(f) = testOps.(f);
    end
    log('i', 'TEST: ops overridden: %s', jsonencode(testOps));
end
cleanup = onCleanup(@() deleteIfExists(ops.fproc));

t0 = tic();
log('i', 'Preprocessing data...');
rez = preprocessDataSub(ops);
rez.ops.spikeTemplatesPath = '';
tPre = toc(t0);
log('i', 'Applying data-shifting ...');
rez = datashift2(rez, 1);
tShift = toc(t0);
log('i', 'Optimizing templates (seed %u)...', seed);
rez = learnAndSolve8b(rez, seed);
tLearn = toc(t0);
rez = autocurate(rez, 'debug', false);
tCurate = toc(t0);

log('i', 'Converting kilosort rez to phy NPY files...');
rezToPhy(rez, char(outDir));
fid = fopen(fullfile(outDir, 'ks25_run.json'), 'w');
fprintf(fid, '%s', jsonencode(struct('seed', seed, 'pipeline_version', npxVersion(), ...
    'matlab', version(), 'gpu', gpu.Name, 'n_batches', rez.ops.Nbatch, ...
    'n_learn_batches', numel(rez.iorderLearn), 'n_templates', size(rez.W, 2), ...
    'seconds', struct('preprocess', tPre, 'datashift', tShift - tPre, ...
    'learn_and_solve', tLearn - tShift, 'autocurate', tCurate - tLearn, 'total', toc(t0))), ...
    'PrettyPrint', true));
fclose(fid);
log('i', 'Done in %.1f min', toc(t0) / 60);
end

function deleteIfExists(f)
% Close any handle still open on f (as runKilosortClustering does), then delete it.
for fid = openedFiles()
    if strcmp(fopen(fid), f), fclose(fid); end
end
if exist(f, 'file'), delete(f); end
end
