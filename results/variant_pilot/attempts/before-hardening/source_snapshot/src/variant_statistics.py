"""Equal-cluster feasibility summaries and intervention-based power planning."""
import math
import numpy as np
from scipy.stats import spearmanr, norm, t


def cluster_summary(values, clusters, repetitions=2000, seed=1731):
    values, clusters=np.asarray(values,dtype=float),np.asarray(clusters)
    if len(values) != len(clusters) or not np.isfinite(values).all():
        raise ValueError("Finite aligned effects required")
    means=np.array([values[clusters==c].mean() for c in np.unique(clusters)])
    if len(means)<2:
        return dict(mean=None,ci_low=None,ci_high=None,clusters=len(means))
    rng=np.random.default_rng(seed)
    draws=means[rng.integers(0,len(means),(repetitions,len(means)))].mean(axis=1)
    low,high=np.quantile(draws,[.025,.975])
    return dict(mean=float(means.mean()),ci_low=float(low),ci_high=float(high),clusters=len(means))


def binding_agreement(scores, effects, clusters, repetitions=2000, seed=1731):
    scores,effects,clusters=np.asarray(scores,dtype=float),np.asarray(effects,dtype=float),np.asarray(clusters)
    if not len(scores)==len(effects)==len(clusters) or not np.isfinite(scores).all() or not np.isfinite(effects).all():
        raise ValueError("Finite aligned model and biological effects required")
    groups=np.unique(clusters)
    x=np.array([scores[clusters==c].mean() for c in groups])
    y=np.array([abs(effects[clusters==c]).mean() for c in groups])
    if len(groups)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return dict(binding_rho=None,binding_ci_low=None,binding_ci_high=None,bootstrap_defined=0)
    rho=float(spearmanr(x,y).statistic)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        index=rng.integers(0,len(groups),len(groups))
        if np.ptp(x[index])>0 and np.ptp(y[index])>0:
            draws.append(float(spearmanr(x[index],y[index]).statistic))
    if len(draws)<.95*repetitions:
        low=high=None
    else:
        low,high=map(float,np.quantile(draws,[.025,.975]))
    return dict(binding_rho=rho,binding_ci_low=low,binding_ci_high=high,bootstrap_defined=len(draws))


def simulate_cluster_power(intervention_cluster_means, minimum_effect, retention=.5,
                           candidates=(16,32,64,128,256,512), repetitions=3000, seed=1731):
    """Resample discovery intervention noise, never unpatched pilot variance."""
    values=np.asarray(intervention_cluster_means,dtype=float)
    if len(values)<8 or not np.isfinite(values).all() or np.std(values,ddof=1)<=0:
        raise ValueError("At least eight variable discovery intervention clusters required")
    if minimum_effect<=0 or not 0<retention<=1 or repetitions<100:
        raise ValueError("Invalid effect, retention or simulation count")
    noise=values-values.mean();rng=np.random.default_rng(seed);rows=[]
    for n in candidates:
        if n<3:
            raise ValueError("Power candidates must contain at least three clusters")
        draws=noise[rng.integers(0,len(noise),(repetitions,n))]+minimum_effect
        se=draws.std(axis=1,ddof=1)/math.sqrt(n)
        threshold=t.ppf(.975,n-1)
        power=float(np.mean(np.abs(draws.mean(axis=1))>threshold*se))
        rows.append(dict(retained_clusters=n,power=power,screen_clusters=math.ceil(n/retention)))
    selected=next((r for r in rows if r["power"]>=.8),None)
    return dict(status="planning only",effect=minimum_effect,cluster_sd=float(values.std(ddof=1)),
        simulations=repetitions,rows=rows,selected=selected,
        limitation="empirical cluster noise; separate biological association power and donor dependence still required")


def select_head(discovery_rows):
    if not discovery_rows:
        raise ValueError("No discovery intervention results")
    if any(not math.isfinite(r["mean"]) for r in discovery_rows):
        raise ValueError("Head selection requires finite effects")
    return min(discovery_rows,key=lambda r:(-r["mean"],r["layer"],r["head"]))
