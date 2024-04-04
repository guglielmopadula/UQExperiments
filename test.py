import numpy as np
import chaospy
import matplotlib.pyplot as plt
#coordinates = np.linspace(1, 1.5, 1000)
coordinates=1+0.5*np.random.rand(500)
coordinates=np.sort(coordinates)

def model_solver(alpha):
    """
    Simple ordinary differential equation solver.

    Args:
        parameters (numpy.ndarray):
            Hyper-parameters defining the model initial
            conditions alpha and growth rate beta.
            Assumed to have ``len(parameters) == 2``.

    Returns:
        (numpy.ndarray):
            Solution to the equation.
            Same shape as ``coordinates``.
    """

    return np.exp(coordinates)*(1+0.1*np.sin(20*coordinates))+alpha.T



#gauss_quads =  chaospy.generate_quadrature(8, alpha, rule="gaussian")
#gauss_nodes,_ =gauss_quads
gauss_nodes=0.1*np.random.randn(1,10)
evals=model_solver(gauss_nodes)
#alpha = chaospy.Normal(0, 0.2)
alpha=chaospy.GaussianKDE(gauss_nodes)
expansion=chaospy.generate_expansion(8,alpha)
model=chaospy.fit_regression(expansion, gauss_nodes, evals)
expected = chaospy.E(model, alpha)
std = chaospy.Std(model, alpha)
up=chaospy.Perc(model,95,alpha).reshape(-1)
down=chaospy.Perc(model,5,alpha).reshape(-1)
fig,ax=plt.subplots()
ax.plot(coordinates,expected,label="predicted") 
ax.fill_between(coordinates, down, up, alpha=0.2)
fig.savefig("test.png")