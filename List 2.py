"""
Created on Wed Oct 26 07:56:42 2022

@author: franc
"""
#import yfinance as yf
import math
from scipy.stats import norm
from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
price_h = yf.download('^BVSP',start="2017-01-01",end="2022-10-09")['Adj Close']
df = pd.Series(price_h)
df.to_csv('BVSP.csv')
'''
bvsp = pd.DataFrame(pd.read_csv('BVSP.csv'))
bvsp['Adj Close'] = np.log(bvsp['Adj Close']/bvsp['Adj Close'].shift(1))
bvspmu = np.mean(bvsp['Adj Close'])
df1 = pd.DataFrame(pd.read_csv('5 stock.csv'))
port = pd.Series(df1['ABEV3.SA'].mul(0))
temp = [0.03395829, 0.06625463, 0.22058755, 0.38663664, 0.29256289]
count = 0
#creating portfolio
for i in df1.columns:
    te = temp[count]
    count += 1
    port = port.add(df1[i].mul(te))
#log return
port=np.log(port/port.shift(1))
mu = np.mean(port)
sigma = np.std(port)
#normalizing
nport = port.subtract(mu)
nport = port.div(sigma)
#plotting histogram + normal
plt.hist(nport, bins = 100, density=True)
x = np.linspace(norm.ppf(0.00001),norm.ppf(0.999999), 100)
plt.plot(x, norm.pdf(x),'r-', lw=1, alpha=0.6, label='norm pdf')
#plotting t distributions
df = 5
x = np.linspace(t.ppf(0.001, df),t.ppf(0.999, df), 200) 
plt.plot(x, t.pdf(x, df),'g-',lw=1)
df = 10
x = np.linspace(t.ppf(0.001, df),t.ppf(0.999, df), 200) 
plt.plot(x, t.pdf(x, df),'b-',lw=1)
df = 50
x = np.linspace(t.ppf(0.001, df),t.ppf(0.999, df), 200) 
plt.plot(x, t.pdf(x, df),'y-',lw=1)
#pdf + cdf
plt.figure()
x = np.linspace(-3*sigma+mu,sigma*3+mu, 100)
plt.plot(x, norm.pdf(x,mu,sigma))
plt.figure()
plt.plot(x, norm.cdf(x,mu,sigma))
#kurtosis and skewness
print("Skew of portfolio: ",port.skew())
print("Kurtosis of portfolio: ",port.kurtosis())
#probability if dist were normal
print("The probability of experiencing a return greater than 3% is ", (1-norm(mu,sigma).cdf(0.03))*100,"%")
#25th percentile + prob of negative number
print("The 25th percentile is ", norm.ppf(0.25,mu,sigma))
print("Probability of negative returns is ", norm.cdf(0,mu,sigma)*100,"%")
#lognormal mean and variance
'''
plt.figure()
eport = port
for i in range(len(eport)):
    eport[i] = math.exp(eport[i])
eport.plot.hist(bins = 100)
'''
print("Lognormal mean is ", math.exp(mu+pow(sigma,2)/2))
print("Lognormal variance is ", math.exp(2*mu+2*pow(sigma,2)) - math.exp(2*mu+pow(sigma,2)))
#hypothesis testing
#finding out whether population mean might be equal to 0 based on sample of last 5 years
confidence = 0.95
alpha = (1-confidence)/2
#unknown population SD so use T distribution (large enough sample so that it shouldn't make a difference)
E = t.ppf(alpha, len(port))*sigma/pow(len(port),1/2)
#null hypothesis portfolio mean is 0,h1 portfolio mean is not equal to 0 
print("Hypothesis test: H0 portfolio mean = 0, H1 portfolio mean != 0")
print("Sample mean minus E ", mu-E)
if mu-E<0 or mu+E>0:
    print("Accept null hypothesis")
else:
    print("Accept alternative hypothesis")
    print("We can therefore state with a 95% confidence level that portfolio returns do not have mean 0. As 0 is outside our confidence interval.")
#bvsp mean
print("Hypothesis test: H0 portfolio mean = BVSP mean, H1 portfolio mean != BVSP mean")
print("bvsp mean ",bvspmu)
if mu-E<bvspmu or mu+E>bvspmu:
    print("Accept null hypothesis")
else:
    print("Accept alternative hypothesis")
    print("We can therefore state with a 95% confidence level that portfolio returns do not have mean of BVSP returns. As BVSP mean is outside our confidence interval.")