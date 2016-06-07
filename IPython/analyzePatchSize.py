# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:09:49 2016

Hexbin plot with marginal distributions

@ Copyright (C) Cristina Gallego, University of Toronto, 2016
"""
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

from query_localdatabase import *

#############################
###### Query local databse to retrieve patches
#############################
print " Querying local databse..."
querylocal = Querylocal()
dfpatches = querylocal.query_allpatches()

# to get all patch size records
patsz = dfpatches.ix[:,1]

# extract patch Height (Ph) and Width (Pw)
Ph = pd.Series([long(ps.split()[0][ps.split()[0].find("(")+1:ps.split()[0].find(",")]) for ps in patsz])
Pw = pd.Series([long(ps.split()[1][ps.split()[1].find(" ")+1:ps.split()[1].find(")")]) for ps in patsz])


#The bivariate analogue of a histogram is known as a “hexbin” plot, because it shows the counts 
#of observations that fall within hexagonal bins. This plot works best with relatively large datasets. 
# source: https://stanford.edu/~mwaskom/software/seaborn/tutorial/distributions.html
g = (sns.jointplot(Ph, Pw, kind="hex")
        .set_axis_labels("Patch Height", "Patch Width"))

# print individual distribution
print(Ph.describe())
print(Pw.describe())
