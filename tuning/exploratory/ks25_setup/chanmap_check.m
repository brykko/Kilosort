addpath('C:\code\github\neuropixels'); Shared.configureStageDependencies('SpikeSort');
metas = {'C:\data\ks4_tuning\25843_roger_2019-09-11\search\source.ap.meta', 'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\source.ap.meta'};
names = {'mec', 'kobe'};
for i = 1:2
    meta = fileread(metas{i});
    tok = regexp(regexp(meta, '~snsShankMap=([^\s]*)', 'once', 'tokens'), '\((\d*):(\d*):(\d*):[0|1]\)', 'tokens');
    shankMap = cellfun(@str2double, vertcat(tok{1}{:})) + 1;
    ielec = (shankMap(:, 3) - 1) * 2 + shankMap(:, 2);
    m = probes.helpers.createKilosortChannelMap(probes.ChannelMap(probes.models.Imec_Neuropixels_3a_opt3, ielec));
    save(['chanmap_' names{i} '.mat'], '-struct', 'm', '-v7');
end
