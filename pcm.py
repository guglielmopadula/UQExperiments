import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import numpyro
from time import time
import chaospy
import matplotlib.pyplot as plt
start=time()
np.random.seed(0)
num_data=5000
sigma=0.1
results=np.zeros((5,6))
x_test=1+0.5*np.random.rand(1000)
x_test=np.sort(x_test)
x_test=x_test.reshape(-1,1)
y_test=np.exp(x_test)*(1+0.1*np.sin(20*x_test))
start=time()
x=1+0.5*np.random.rand(num_data)
x=x.reshape(-1)
x=np.sort(x)
y=np.exp(x)*(1+0.1*np.sin(20*x))+sigma*np.random.randn(num_data)
#This step is necessary as PCE wants a complete sample in every x.
x=x.reshape(500,-1)
x_old=x.copy()
y_old=y.copy()
x_old=x_old.reshape(-1)
x=np.mean(x,axis=1)
x=x.reshape(1,500)
y=y.reshape(500,10).T #shape (samples,dim)
#Polynomial chaos expansion wants some samples of the parameters for each x wants low dimensional parametrization, 
#as we have a low number of samples, we can use PCA.
#parameters=y-np.mean(y,axis=0)
#pca=PCA(n_components=10)
#pca.fit(parameters)
#parameters=pca.transform(parameters)
#Or we can use Isomap, autoencoders....
isomap=Isomap(n_components=5)
parameters=isomap.fit_transform(X=y)
distribution = chaospy.GaussianKDE(parameters.T)
expansion = chaospy.generate_expansion(1, distribution, rule="cholesky")
model=chaospy.fit_regression(expansion, parameters.T, y)
expected = chaospy.E(model,distribution)
up=chaospy.Perc(model,95,distribution).reshape(-1)
down=chaospy.Perc(model,5,distribution).reshape(-1)
x=x.reshape(-1)
expected=expected.reshape(-1)
fig,ax=plt.subplots()
ax.scatter(x_old,y_old.reshape(-1),label="noisy") 
ax.plot(x,expected,label="predicted") 
ax.fill_between(x, down, up, alpha=0.2)
ax.plot(x_old,np.exp(x_old)*(1+0.1*np.sin(20*x_old)),label="true")

ax.legend()
fig.savefig('pcm.png')
print(time()-start)
print(np.mean((up-down)/expected))
print(np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))-expected)/np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))))
end=time()
