import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import numpyro
from time import time
import chaospy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
np.random.seed(0)
start=time()
num_data=5000
sigma=0.1
x=1+0.5*np.random.rand(num_data)
x=np.sort(x)
y=np.exp(x)*(1+0.1*np.sin(20*x))+sigma*np.random.randn(num_data)
gp=GaussianProcessRegressor(RBF(1.0) + WhiteKernel(noise_level=0.5))
gp.fit(x.reshape(-1,1), y.reshape(-1,1))
y_pred, y_pis = gp.predict(x.reshape(-1,1),return_std=True)
x=x.reshape(-1)
down=y_pred-1.64*y_pis
up=y_pred+1.64*y_pis
down=down.reshape(-1)
up=up.reshape(-1)
y_pred=y_pred.reshape(-1)
fig,ax=plt.subplots()
#ax.plot(x,y,label="noised")
ax.plot(x,y_pred,label="predicted")
ax.fill_between(x, down, up, alpha=0.2)
ax.plot(x,np.exp(x)*(1+0.1*np.sin(20*x)),label="true")

ax.legend()
fig.savefig('gpr.png')
y_pred=y_pred.reshape(-1)
print(time()-start)
print(np.mean((up-down)/y_pred))
print(np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))-y_pred)/np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))))