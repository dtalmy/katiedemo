import ODElib
import scipy
import numpy as np
import pylab as py
import pandas as pd

###################################################
# read in data
###################################################

# read whole dataset
master_df = pd.read_csv('../data/tradeoffdat.csv')

# calculate mean abundance and std
master_df['log_abundance'] = np.mean(np.log(np.r_[[master_df[i] for i in ['r1','r2','r3']]]),axis=0)
master_df['abundance'] = np.mean(np.r_[[master_df[i] for i in ['r1','r2','r3']]],axis=0)
master_df['log_sigma'] = np.std(np.log(np.r_[[master_df[i] for i in ['r1','r2','r3']]]),axis=0)

# select a single control curve for fitting
zerodmspdat = master_df[master_df['treatment']=='0DMSP']
h0dmsp = zerodmspdat[zerodmspdat['control']==True]
single= h0dmsp[h0dmsp['organism']=='H']

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

# log-transformed priors
alpha_prior=ODElib.parameter(stats_gen=scipy.stats.lognorm,
                      hyperparameters={'s':1,'scale':1e-6})

# define initial conditions
H0 = single[single['organism']=='H']['abundance'].iloc[0]
N0 = single[single['organism']=='H']['abundance'].iloc[-1] - H0

# initiate class with no infection states
H1=ODElib.ModelFramework(ODE=holling_one,
                          parameter_names=['alpha'],
                          state_names = ['N','H'],
                          dataframe=single,
                          alpha = alpha_prior.copy(),
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
ax[0].plot(H1.times,mod['N'],label='initial guess')
ax[1].plot(H1.times,mod['H'],label='initial guess')

###################################################
# run model
###################################################

# call the MCMC algorithm
posteriors = H1.MCMC(iterations_per_chain=1000,
                       cpu_cores=2,fitsurvey_samples=1000,sd_fitdistance=20.0)

# set optimal parameters
im = posteriors.loc[posteriors.chi==min(posteriors.chi)].index[0]
pdic = posteriors.loc[im][H1.get_pnames()].to_dict()
H1.set_parameters(**posteriors.loc[im][H1.get_pnames()].to_dict())

# run the model again, now with fitted parameters
mod = H1.integrate()

# plot fitted model
ax[0].plot(H1.times,mod['N'],label='fitted')
ax[1].plot(H1.times,mod['H'],label='fitted')

# legend
l = ax[1].legend()
l.draw_frame(False)

py.show()


