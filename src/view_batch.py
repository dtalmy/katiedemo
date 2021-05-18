import pandas as pd
import numpy as np
import pylab as pl

# reading master dataset
master_df = pd.read_csv('../data/tradeoffdat.csv')

# select a random subset for plotting
zerodmspdat = master_df[master_df['treatment']=='0DMSP']
h0dmsp = zerodmspdat[zerodmspdat['control']==True]
single= h0dmsp[h0dmsp['org']=='H']

# now plot
f,ax = pl.subplots()
ax.plot(single.time,single.r1,'-o')
ax.plot(single.time,single.r2,'-o')
ax.plot(single.time,single.r3,'-o')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Abundance (ml-1)')

# savefig
f.savefig('../figures/demodat')
pl.show()
