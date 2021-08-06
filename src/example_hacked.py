import ODElib
import scipy
import numpy as np
import pylab as py
import pandas as pd

###################################################
# read in data
###################################################

# read whole dataset
single = pd.read_csv('../data/for_demo.csv')

###################################################
# define model
###################################################

def holling_one(y,t,ps):
    alpha=ps[0]
    N,H = y[0],y[1]
    dNdt = -alpha*N*H
    dHdt = alpha*N*H
    return [dNdt,dHdt]

###################################################
# initialize model
###################################################

# define initial conditions
H0 = single[single['organism']=='H']['abundance'].iloc[0]
N0 = single[single['organism']=='H']['abundance'].iloc[-1] - H0

# log-transformed priors
alpha_prior=ODElib.parameter(stats_gen=scipy.stats.lognorm,
                      hyperparameters={'s':1,'scale':1e-6})

N0_prior=ODElib.parameter(stats_gen=scipy.stats.lognorm,
                      hyperparameters={'s':1,'scale':N0})

# initialize the class for fitting
H1=ODElib.ModelFramework(ODE=holling_one,
                          parameter_names=['alpha','N0'], 
                          state_names = ['N','H'],
                          dataframe=single,
                          alpha = alpha_prior.copy(),
                          N0 = N0_prior.copy(),
                          t_steps=288,
                          H = H0,
                          N = N0
                         )

###################################################
# visualize initial parameter guess
###################################################

# setup figure
f,ax = py.subplots(1,2,figsize=[9,4.5])
ax[0].set_xlabel('Time (days)')
ax[1].set_xlabel('Time (days)')
ax[0].set_ylabel('Nutrients (cellular equivalents ml$^{-1}$)')
ax[1].set_ylabel('Cells ml$^{-1}$')
ax[0].semilogy()
ax[1].semilogy()

# plot data
ax[1].errorbar(H1.df.loc['H']['time'],
                            H1.df.loc['H']['abundance'],
                            yerr=H1._calc_stds('H')
                            )

# integrate the model once with initial parameter guess
mod = H1.integrate()

# plot model initial guess
ax[0].plot(H1.times,mod['N'],label='initial guess',c='r')
ax[1].plot(H1.times,mod['H'],label='initial guess',c='r')

###################################################
# do fitting and plot fitted model
###################################################

# provide reasonable guesses for the mcmc algorithm
chain_inits = pd.DataFrame({'alpha':[1e-6]*2,'N0':[N0]*2})

# call the MCMC algorithm to fit parameters
posteriors = H1.MCMC(chain_inits=chain_inits,iterations_per_chain=1000,
                       cpu_cores=2,fitsurvey_samples=1000,sd_fitdistance=20.0)

# set optimal parameters
im = posteriors.loc[posteriors.chi==min(posteriors.chi)].index[0]
H1.set_parameters(**posteriors.loc[im][H1.get_pnames()].to_dict())
H1.set_inits(**{'N':posteriors.loc[im][H1.get_pnames()].to_dict()['N0']})

# run the model again, now with fitted parameters
mod = H1.integrate()

# plot fitted model
ax[0].plot(H1.times,mod['N'],label='fitted',c='g')
ax[1].plot(H1.times,mod['H'],label='fitted',c='g')

# legend
l = ax[1].legend()
l.draw_frame(False)

py.show()

# save output
f.savefig('../figures/batch_curve_fitting')

