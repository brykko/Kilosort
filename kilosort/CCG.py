from numba import njit
import numpy as np
from scipy.stats import norm
import torch
from torch.nn.functional import conv1d
import math 
from tqdm import trange 

@njit(cache=True)
def compute_CCG(st1, st2, tbin = 1/1000, nbins = 500):

    st1 = np.sort(st1)
    st2 = np.sort(st2)

    dt = nbins * tbin
    T = np.maximum(st1.max(), st2.max()) - np.minimum(st1.min(), st2.min())

    ilow = 0
    ihigh = 0
    j = 0

    K = np.zeros(2*nbins+1,)
    while j<len(st2):
        while (ihigh<len(st1)) and (st1[ihigh] <= st2[j]+dt):
            ihigh += 1
        while (ilow<len(st1))  and (st1[ilow] <= st2[j]-dt):
            ilow += 1
        if ilow >=len(st1):
            break
        if st1[ilow]>st2[j]+dt:
            j += 1
            continue;
        for k in range(ilow, ihigh):
            ibin = int(np.round((st2[j] - st1[k])/tbin))
            K[ibin+nbins] += 1
        j += 1
    return K, T

#@njit()
def CCG_metrics(st1, st2, K, T, nbins=None, tbin=None):
    irange1 = np.hstack((np.arange(1,nbins//2), np.arange(3*nbins//2, 2*nbins)))
    irange2 = np.arange(nbins-50, nbins-10)
    irange3 = np.arange(nbins+10, nbins+50)

    R00 = K[irange1].sum() / (len(irange1) * tbin * len(st1) * len(st2) /T)
    R1 = K[irange2].sum() / (len(irange2) * tbin * len(st1) * len(st2) /T)
    R2 = K[irange3].sum() / (len(irange3) * tbin * len(st1) * len(st2) /T)
    R01 = np.maximum(R1, R2)

    Q00 = np.maximum(K[irange2].mean(), K[irange3].mean())
    Q00 = np.maximum(Q00, K[irange1].mean())

    a = K[nbins]
    K[nbins] = 0

    Ri = np.zeros(10,)
    Qi = np.zeros(10,)
    for i in range(1,11):
        irange = np.arange(nbins-i, nbins+i+1)
        Ri0 = K[irange].sum() / (2*i*tbin*len(st1)*len(st2)/T)
        Ri[i-1] = Ri0

        n = K[irange].sum()/2
        lam = Q00 * i

        p = 1/2 * (1 + math.erf((n-lam)/(1e-10 + 2*lam)**.5))

        Qi[i-1] = p

    K[nbins] = a

    # NOTE: The variable names R12 and Q12 are reversed here compared to past
    #       versions of Kilosort, but the code remains unchanged. This change
    #       was made to be consistent with the methods in the Kilosort4 paper.
    R12 = np.min(Ri) / (1e-10 + np.maximum(R00, R01))
    Q12 = np.min(Qi)

    #print('%4.2f, %4.2f, %4.2f'%(R00, Q12, R12))
    return R12, Q12, Q00

def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000, acg_threshold=0.2,
              ccg_threshold=0.25, use_shape_criterion=False):
    # NOTE: The default `acg_threshold=0.2` is different from the value of 0.1
    #       used for the Kilosort4 paper. We felt this better reflects common
    #       practice for determining 'good' units, but you can set
    #       `acg_threshold=0.1` in your run settings for stricter criteria.
    if st2 is None:
        st2 = st1.copy()
    K , T= compute_CCG(st1, st2, nbins = nbins, tbin = tbin)
    R12, Q12, Q00 = CCG_metrics(st1, st2, K, T,  nbins = nbins, tbin = tbin)
    is_refractory    = R12<acg_threshold  and (Q12<.2)#  or Q00<.25)
    cross_refractory = R12<ccg_threshold and (Q12<.05)# or Q00<.25)
    # Optional shape criterion (two different units only): a pair counts as
    # refractory only if its CCG also has the shape of both ACGs.
    if use_shape_criterion and (is_refractory or cross_refractory):
        if not same_shape(st1, st2, K, nbins=nbins, tbin=tbin):
            is_refractory, cross_refractory = False, False
    return is_refractory, cross_refractory, R12

def similarity(Wall, W, nt=61):
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])
    mu = (Wall**2).sum((1,2), keepdims=True)**.5
    Wnorm = Wall / (1e-6 + mu)
    UtU = torch.einsum('ilk, jlm -> ijkm',  Wnorm, Wnorm)
    similar_templates = torch.einsum('ijkm, kml -> ijl', UtU.cpu(), WtW.cpu()).numpy()
    similar_templates = similar_templates.max(axis=-1)
    return similar_templates

def refract(iclust2, st0, acg_threshold=0.2, ccg_threshold=0.25):
    
    Nfilt = iclust2.max()+1

    is_refractory    = np.zeros(Nfilt, )
    cross_refractory = np.zeros(Nfilt, )
    R12 = np.zeros(Nfilt, )

    # Spike indices of each cluster, in their original order (a stable sort),
    # instead of a boolean mask over all spikes per cluster.
    order = np.argsort(iclust2, kind='stable')
    bounds = np.concatenate(([0], np.cumsum(np.bincount(iclust2, minlength=Nfilt))))

    for kk in range(Nfilt):
        st1 = st0[order[bounds[kk]:bounds[kk+1]]]

        if (len(st1) > 10) and ((st1.max() - st1.min()) != 0):
            is_refractory[kk], cross_refractory[kk], R12[kk] = check_CCG(
                st1, acg_threshold=acg_threshold, ccg_threshold=ccg_threshold
                )

    return is_refractory, R12


SHAPE_THRESHOLD = 0.5

def same_shape(st1, st2, K=None, nbins=500, tbin=1/1000):
    """True if the CCG of st1 and st2 has the shape of both ACGs (Pearson r >
    SHAPE_THRESHOLD, see CCG_ACG_similarity). Guards against merging units
    whose CCG has a broad trough from anti-phase firing, not a refractory dip."""
    r, _ = CCG_ACG_similarity(st1, st2, tbin=tbin, nbins=nbins, K=K)
    return r > SHAPE_THRESHOLD

def CCG_ACG_similarity(st1, st2, tbin = 1/1000, nbins = 500, max_lag=0.050, K=None):
    """Similarity of the CCG's shape to each ACG's within +-max_lag: the smaller
    of the two CCG_corr values. K is the CCG of st1 and st2 if already computed."""
    if K is None:
        K, _ = compute_CCG(st1, st2, tbin=tbin, nbins=nbins)
    A1, _ = compute_CCG(st1, st1, tbin=tbin, nbins=nbins)
    A2, _ = compute_CCG(st2, st2, tbin=tbin, nbins=nbins)
    lags = np.arange(-nbins, nbins+1) * tbin

    # compute_CCG() counts each spike against itself at lag 0; remove that
    i0 = np.where(lags==0)
    A1[i0] -= len(st1)
    A2[i0] -= len(st2)

    correlograms = {'CCG': K, 'ACG1': A1, 'ACG2': A2, 'lags': lags}
    similarity = np.min((CCG_corr(lags, K, A1, max_lag), CCG_corr(lags, K, A2, max_lag)))
    return similarity, correlograms

def CCG_corr(T, C1, C2, max_lag):
    """Pearson r between two smoothed correlograms (Gaussian, SD 4 bins) on each
    side of zero lag within +-max_lag; the larger of the two sides."""
    TOL = 1e-6 # numerical tolerance to prevent rounding errors

    x = np.arange(-10, 11)
    f = norm.pdf(x, loc=0, scale=4)
    f = f / np.sum(f)
    C1 = np.convolve(C1, f, mode='same')
    C2 = np.convolve(C2, f, mode='same')

    viR = np.abs(T) <= (max_lag + TOL)
    vleft = viR & (T < 0)
    vright = viR & (T > 0)

    # Too few points on a side gives r = 0 for that side
    r_l = np.corrcoef(C1[vleft], C2[vleft])[0, 1] if np.sum(vleft) > 1 else 0.0
    r_r = np.corrcoef(C1[vright], C2[vright])[0, 1] if np.sum(vright) > 1 else 0.0
    return max(r_l, r_r)
