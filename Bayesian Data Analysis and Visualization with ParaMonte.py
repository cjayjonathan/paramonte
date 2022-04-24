#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Installing the Paramonte library
get_ipython().system('pip install paramonte')


# In[5]:


# Import necessary libraries

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[7]:


get_ipython().system('pip install paramonte')


# In[8]:


import paramonte as pm


# In[11]:


# Loading and Analyzing Data

df=pd.read_csv('Datasets/test.csv')
fig = plt.figure  ( figsize = (4,5)
                  , dpi = 100
                  )
ax = fig.add_subplot(1,1,1)
ax.scatter( df['x']
          , df['y']
          , color = "red" 
          , s = 5 
          )
plt.show()


# In[12]:


# Application of Log to Data Points
from scipy.stats import norm
logX = np.log(X)
logY = np.log(y)
 
def getLogLike(param):  
    mean = param[0] + param[1] * logX    
    logProbDensities = norm.logpdf(logY, loc = mean, scale = np.exp(param[2])) 
    return np.sum(logProbDensities)


# In[13]:


# Building the ParaMonte Model

para = pm.ParaDRAM()
para.spec.overwriteRequested = True 
para.spec.outputFileName = "./regression_powerlaw" 
para.spec.randomSeed = 100 
para.spec.variableNameList = ["intercept", "slope", "logSigma"]
para.spec.chainSize = 1000 
para.runSampler( ndim = 3 
               , getLogFunc = getLogLike 
               )


# In[14]:


# Let’s visualize the chain and the samples created by the sampler

chain.plot.scatter( ycolumns = "AdaptationMeasure"
                  , ccolumns = [] 
                  )
chain.plot.scatter.currentFig.axes.set_ylim([1.e-5,1])
chain.plot.scatter.currentFig.axes.set_yscale("log")


# In[15]:


sample = para.readSample(renabled = True)[0]
for colname in sample.df.columns:
    sample.plot.line.ycolumns = colname
    sample.plot.line()
    sample.plot.line.currentFig.axes.set_xlabel("MCMC Count")
    sample.plot.line.currentFig.axes.set_ylabel(colname)
    sample.plot.line.savefig( fname = "/traceplot_" + colname )
 
 
for colname in sample.df.columns:
    sample.plot.histplot(xcolumns = colname)
    sample.plot.histplot.currentFig.axes.set_xlabel(colname)
    sample.plot.histplot.currentFig.axes.set_ylabel("MCMC Count")
    sample.plot.histplot.savefig( fname = "/histogram_" + colname )


# In[16]:


# As we can observe the distribution is almost normal for all the variables. 
# But the Log function distribution is skewed to the right. 
# Let’s see how this is affecting the linear relationship between the dependent variable and independent variable.

values = np.linspace(0,100,101)
yvalues = np.exp(sample.df["intercept"].mean()) * xvalues ** sample.df["slope"].mean()
 
fig = plt.figure(figsize = (4.5,4), dpi = 100)
ax = fig.add_subplot(1,1,1)
 
ax.plot(xvalues, yvalues, "b")
ax.scatter(X, y, color = "red", s = 5)


# In[17]:


# We can visualize them together to understand the level of uncertainty in our best-fit regression.

values = np.linspace(0,100,101)
 
fig = plt.figure(figsize = (4.5,4), dpi = 100)
ax = fig.add_subplot(1,1,1)
 
 
first = 0
last = 300
slopes = sample.df["slope"].values[first:last]
intercepts = sample.df["intercept"].values[first:last]
 
for slope, intercept in zip(slopes, intercepts):
    yvalues = np.exp(intercept) * xvalues ** slope
    ax.plot( xvalues
           , yvalues
           , "black" 
           , alpha = 0.04 
           )
 
ax.scatter( X, y
          , color = "red"
          , s = 5
          , zorder = 100000
          )


# In[ ]:




