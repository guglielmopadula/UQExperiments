import numpy as np
import numpyro
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoNormal, AutoIAFNormal
from time import time
import numpyro.distributions as dist
from svimodel import SVIModel 
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from mapie.regression import MapieQuantileRegressor, MapieRegressor
import matplotlib.pyplot as plt
start=time()
np.random.seed(0)
num_data=5000
sigma=0.1
results=np.zeros((5,6))
kernel=RBF(1.0) + WhiteKernel(noise_level=0.5)
x_test=1+0.5*np.random.rand(1000)
x_test=np.sort(x_test)
x_test=x_test.reshape(-1,1)
y_test=np.exp(x_test)*(1+0.1*np.sin(20*x_test))
x=1+0.5*np.random.rand(num_data)
x=np.sort(x)
y=np.exp(x)*(1+0.1*np.sin(20*x))+sigma*np.random.randn(num_data)
gp=KernelRidge(kernel=kernel,alpha=1e-10)
mp=MapieRegressor(gp, method='naive',cv='split',test_size=0.1)
mp.fit(x.reshape(-1,1), y)
y_pred, y_pis = mp.predict(x.reshape(-1,1), alpha=0.1)
y_pis=y_pis[:,:,0]
x=x.reshape(-1)
down=y_pis[:,0].reshape(-1)
up=y_pis[:,1].reshape(-1)
fig,ax=plt.subplots()
#ax.plot(x,y,label="noised")
ax.plot(x,y_pred,label="predicted")
ax.fill_between(x, down, up, alpha=0.2)
ax.plot(x,np.exp(x)*(1+0.1*np.sin(20*x)),label="true")

ax.legend()
fig.savefig('gpr_split_conformal.png')
y_pred=y_pred.reshape(-1)
print(time()-start)
print(np.mean((up-down)/y_pred))
print(np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))-y_pred)/np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))))