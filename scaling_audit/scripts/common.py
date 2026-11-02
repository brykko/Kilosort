def refr_train(rate, T, dead=0.002):
    n = int(rate*T*1.5)+100
    isi = dead + rng.exponential(1/rate-dead, n)
    t = np.cumsum(isi); return t[t<T]
def contaminated(rate, c, T):
    u = refr_train(rate*(1-c), T)
    p = rng.uniform(0, T, rng.poisson(rate*c*T))
    return np.sort(np.concatenate((u,p)))
