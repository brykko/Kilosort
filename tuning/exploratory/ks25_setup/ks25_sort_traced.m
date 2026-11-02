function ks25_sort_traced(fbinary, metaFile, outDir, seed, testOps)
%KS25_SORT_TRACED ks25_sort.m with autocurate's steps run one by one and the spikes of each stage saved.
%   Same setup, preset and steps as tuning/ks25_sort.m; autocurate (two cycles of find_merges,
%   splitAllClusters and set_cutoff, then a final find_merges, as kilosort_v2.5/postProcess/autocurate.m
%   with its defaults) is expanded here so that rez.st3 can be saved after every step: STAGES.mat in
%   OUTDIR holds, per stage, each spike's sample (t), template (clu), amplitude (amp) and template-fit
%   score (score, st3(:,4), the quantity set_cutoff thresholds), plus set_cutoff's per-template
%   thresholds (Ths). KS2.5 is not deterministic, so this sort is compared with its own stages only.
%
%   KS25_SORT_TRACED(..., TESTOPS) overrides ops fields with those of the struct TESTOPS (smoke tests only).
%
%   Run with MATLAB R2025a: matlab -batch "ks25_sort_traced('<bin>', '<meta>', '<outdir>', 1)"

npxRoot = 'C:\code\github\neuropixels';
addpath(npxRoot);
Shared.configureStageDependencies('SpikeSort');
parallel.gpu.enableCUDAForwardCompatibility(true);
gpu = gpuDevice();
SpikeSort.setKilosortPath(2.5);

if ~isfolder(outDir), mkdir(outDir); end
log = Shared.initializeLogging(char(outDir), SpikeSort.Consts.LOG_FILE_NAME);
log('i', 'KS2.5 standard sort (traced) of "%s", seed %u, pipeline version %s, GPU %s', ...
    fbinary, seed, npxVersion(), gpu.Name);

meta = fileread(metaFile);
tok = regexp(regexp(meta, '~snsShankMap=([^\s]*)', 'once', 'tokens'), ...
    '\((\d*):(\d*):(\d*):[0|1]\)', 'tokens');
shankMap = cellfun(@str2double, vertcat(tok{1}{:})) + 1;
nSavedChans = str2double(regexp(meta, 'nSavedChans=(\d*)', 'once', 'tokens'));
assert(size(shankMap, 1) == 384 && nSavedChans == 385, 'expected a 3A option 3 recording');
ielec = (shankMap(:, 3) - 1) * 2 + shankMap(:, 2);
chanMap = probes.helpers.createKilosortChannelMap( ...
    probes.ChannelMap(probes.models.Imec_Neuropixels_3a_opt3, ielec));
chanMapFile = fullfile(outDir, SpikeSort.Consts.CHANNELMAP_FILE_NAME);
save(chanMapFile, '-struct', 'chanMap', '-v7');

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
end
cleanup = onCleanup(@() deleteIfExists(ops.fproc));

t0 = tic();
rez = preprocessDataSub(ops);
rez.ops.spikeTemplatesPath = '';
rez = datashift2(rez, 1);
rez = learnAndSolve8b(rez, seed);
S = struct();
S.extracted = snap(rez);

% autocurate(rez, 'debug', false), step by step
rez.ops.debug = false;
rez = find_merges(rez, "auto", []);             S.merge1 = snap(rez);
rez = splitAllClusters(rez, "auto", []);        S.split1 = snap(rez);
rez = set_cutoff(rez, 1);                       S.cut1 = snap(rez);
assert(rez.ops.repeatCuration, 'the standard preset repeats curation');
rez = find_merges(rez, "auto", []);             S.merge2 = snap(rez);
rez = splitAllClusters(rez, "auto", []);        S.split2 = snap(rez);
rez = set_cutoff(rez, 0);                       S.cut2 = snap(rez);
rez = find_merges(rez, "auto", []);             S.merge3 = snap(rez);
rez = structfun(@gather, rez, "uni", 0);
rez.curateInfo = struct("method", "auto", "date", string(datetime), "pipelineVersion", npxVersion);
Ths = cellfun(@(x) double(x(:)'), rez.Ths, 'uni', 0);
for f = string(fieldnames(S))'
    log('i', 'stage %s: %u spikes', f, numel(S.(f).t));
end
save(fullfile(outDir, 'stages.mat'), '-struct', 'S', '-v7');
save(fullfile(outDir, 'set_cutoff.mat'), 'Ths', '-v7');

rezToPhy(rez, char(outDir));
log('i', 'Done in %.1f min', toc(t0) / 60);
end

function s = snap(rez)
st3 = gather(rez.st3);
s = struct('t', st3(:, 1), 'clu', int32(st3(:, 2)), 'amp', single(st3(:, 3)), 'score', single(st3(:, 4)));
end

function deleteIfExists(f)
for fid = openedFiles()
    if strcmp(fopen(fid), f), fclose(fid); end
end
if exist(f, 'file'), delete(f); end
end
