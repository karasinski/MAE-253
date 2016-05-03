##import numpy as np

##df = pd.read_csv('astro_degrees.csv')
##data = df.degree.values

###data = np.random.random(10000)
##distributions = [st.laplace, st.norm, st.powerlaw]
##mles = []

##for distribution in distributions:
##    pars = distribution.fit(data)
##    mle = distribution.nnlf(pars, data)
##    mles.append(mle)

##results = [(distribution.name, mle) for distribution, mle in zip(distributions, mles)]
##best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
##print('Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1]))


#import pandas as pd
#import matplotlib.pyplot as plt
#import scipy
#import scipy.stats
#import scipy.stats as st

#df = pd.read_csv('astro_degrees.csv')
#size = len(df)
##x = df.x.values
#x = list(range(df.groupby('degree').count().reset_index().degree.values[-1] + 1))
#y = df.degree.values
#bins = df.degree.max()

#df.groupby('degree').count().reset_index().plot(kind='scatter', x='degree', y='x')
##h = plt.hist(y, bins=range(1, bins), color='w')

#distributions = [st.norm, st.expon, st.powerlaw]
##distributions = [st.norm, st.expon, st.exponpow, st.lognorm, st.powerlaw]
#mles = []
#for distribution in distributions:
#    param = distribution.fit(y)
#    pdf_fitted = distribution.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
#    plt.plot(pdf_fitted, label=distribution.name)
#    mle = distribution.nnlf(param, y)
#    mles.append(mle)

#results = [(distribution.name, mle) for distribution, mle in zip(distributions, mles)]
#best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
#print('Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1]))

#plt.xlim(0, bins)
#plt.ylim(0, h[0].max() + 1)
#plt.legend(loc='upper right')
#plt.show()

#import igraph as ig
#import louvain


#G = ig.Graph.Read_Ncol('part-r-00000')
#part = louvain.find_partition(G, method='Modularity', weight='weight');
#cg = part.cluster_graph(combine_vertices='max', combine_edges='sum')

#part2 = louvain.find_partition(cg, method='Modularity', weight='weight');
#cg2 = part2.cluster_graph(combine_vertices='max', combine_edges='sum')


import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

oo = pd.read_csv('oo.csv')

## Power Law ###################################################################
#xdata = oo['degree'].values
#ydata = oo['count'].values

#logx = np.log10(xdata)
#logy = np.log10(ydata)

## define our (line) fitting function
#fitfunc = lambda p, x: p[0] + p[1] * x
#errfunc = lambda p, x, y: (y - fitfunc(p, x))

#pinit = [1.0, 1.0]
#out = optimize.leastsq(errfunc, pinit,
#                       args=(logx, logy), full_output=1)
#pfinal = out[0]

#index = pfinal[1]
#amp = 10.0**pfinal[0]

#powerlaw = lambda x, amp, index: amp * (x**index)
#plt.plot(xdata, powerlaw(xdata, amp, index), label='Power Law')     # Fit
#rmse = ((ydata - powerlaw(xdata, amp, index)) ** 2).mean()**0.5
#print('Power law RMSE {:2.2f}'.format(rmse))


# Power Law ###################################################################
xdata = oo['degree'].values
ydata = oo['count'].values

# define our (line) fitting function
fitfunc = lambda p, x: p[0] * (x**p[1])
errfunc = lambda p, x, y: (y - fitfunc(p, x))

pinit = [1.0, 1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(xdata, ydata), full_output=1)
pfinal = out[0]

index = pfinal[1]
amp = 10.0**pfinal[0]

plt.plot(xdata, fitfunc(pfinal, xdata), label='Power Law')     # Fit
rmse = ((ydata - fitfunc(pfinal, xdata)) ** 2).mean()**0.5
print('Power law RMSE {:2.2f}'.format(rmse))

# Gaussian ####################################################################
def gauss_function(p, x):
    return p[0] * np.exp(-(x-p[1])**2/(2*p[2]**2))

fitfunc = lambda p, x: gauss_function(p, x)
pinit = [1.0, 1.0, 1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(xdata, ydata), full_output=1)

pfinal = out[0]

amp = pfinal[0]
mean = pfinal[1]
std = pfinal[2]

powerlaw = lambda x, amp, index: amp * (x**index)

plt.plot(xdata, gauss_function(pfinal, xdata), label='Gaussian')     # Fit
rmse = ((ydata - gauss_function(pfinal, xdata)) ** 2).mean()**0.5
print('Gaussian RMSE {:2.2f}'.format(rmse))

# Exponential #################################################################
def exponential(p, x):
    return p[0] * p[1]**(x)

# define our (line) fitting function
fitfunc = lambda p, x: exponential(p, x)
pinit = [1.0, 1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(xdata, ydata), full_output=1)
pfinal = out[0]

plt.plot(xdata, exponential(pfinal, xdata), label='Exponential')     # Fit
rmse = ((ydata - exponential(pfinal, xdata)) ** 2).mean()**0.5
print('Exponential RMSE {:2.2f}'.format(rmse))

plt.errorbar(xdata, ydata, fmt='k.')  # Data
plt.xlabel('Degree')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.savefig('fit_comparison.pdf')
plt.show()
