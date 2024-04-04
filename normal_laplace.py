import numpy as np
import numpyro
from time import time
from svimodel import SVIModel 
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


x=1+0.5*np.random.rand(num_data)
x=x.reshape(-1)
x=np.sort(x)
y=np.exp(x)*(1+0.1*np.sin(20*x))+sigma*np.random.randn(num_data)
mp = SVIModel()
mp.fit(x.reshape(-1,1), y.reshape(-1,1), 10000)

'''
y_pred, y_pis = mp.predict(x_test)
x_test=x_test.reshape(-1)
down=y_pis[0].reshape(-1)
up=y_pis[1].reshape(-1)
fig,ax=plt.subplots()
ax.plot(x_test,y_pred,label="predicted")
ax.fill_between(x_test, down, up, alpha=0.2)
ax.plot(x_test,y_test,label="true")
ax.legend()
fig.savefig('split_prediction.png')
'''


y_pred, y_pis = mp.predict(x.reshape(-1,1))
x_test=x_test.reshape(-1)
down=y_pis[0].reshape(-1)
up=y_pis[1].reshape(-1)
y_pred=y_pred.reshape(-1)
fig,ax=plt.subplots()
ax.scatter(x,y,label="noisy")
ax.plot(x,y_pred,label="predicted") 
ax.fill_between(x, down, up, alpha=0.2)
ax.plot(x,np.exp(x)*(1+0.1*np.sin(20*x)),label="true")

ax.legend()
fig.savefig('normal_laplace.png')
y_pred=y_pred.reshape(-1)
print(time()-start)
print(np.mean((y_pis[1]-y_pis[0])/y_pred))
print(np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))-y_pred)/np.linalg.norm(np.exp(x)*(1+0.1*np.sin(20*x))))