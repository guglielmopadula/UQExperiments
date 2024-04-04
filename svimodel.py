import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random
from jax import jit
import numpyro
from numpyro import handlers
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
matplotlib.use("Agg")  # noqa: E402

NUM_SAMPLES=200
NUM_CHAINS=4

# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)


class SVIModel():

    def __init__(self,hidden_prior=dist.Normal(jnp.zeros((100, 100)), jnp.ones((100, 100))),start_prior=dist.Normal(jnp.zeros((1, 100)), jnp.ones((1, 100))),
                 final_prior=dist.Normal(jnp.zeros((100, 1)), jnp.ones((100, 1))), guide_fun=AutoLaplaceApproximation):
        self.start_prior=start_prior
        self.hidden_prior=hidden_prior
        self.final_prior=final_prior
        self.guide_fun=guide_fun
    def model(self,X, Y):
        N, D_X = X.shape

        # sample first layer (we put unit normal priors on all weights)
        w1 = numpyro.sample("w1", self.start_prior)
        z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations

        # sample second layer
        w2 = numpyro.sample("w2", self.hidden_prior)
        z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations

        # sample second layer
        w3 = numpyro.sample("w3", self.hidden_prior)
        z3 = nonlin(jnp.matmul(z2, w3))  # <= second layer of activations

        # sample second layer
        w4 = numpyro.sample("w4", self.hidden_prior)
        z4 = nonlin(jnp.matmul(z3, w4))  # <= second layer of activations


        # sample final layer of weights and neural network output
        w5 = numpyro.sample("w5", self.final_prior)
        sigma=numpyro.sample("sigma", dist.Gamma(1,1))
        z5 = jnp.matmul(z4, w5)  # <= output of the neural network

        numpyro.sample("Y", dist.Normal(z5, sigma), obs=Y)

    # helper function for HMC inference
    def run_inference(self,model, rng_key, X, Y,num_epochs):
        start = time.time()
        self.guide=self.guide_fun(model)
        optimizer = numpyro.optim.Adam(step_size=0.0005)
        svi = SVI(model, self.guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(rng_key, num_epochs, X,Y)
        self.params = svi_result.params
        self.predictive = Predictive(model, guide=self.guide, params=self.params, num_samples=10000)

    def inner_predict(self,model, rng_key, samples, X):
        model = handlers.substitute(handlers.seed(model, rng_key), samples)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = handlers.trace(model).get_trace(X=X, Y=None)
        return model_trace["Y"]["value"]

    def fit(self,X,Y, num_epochs):
        rng_key=random.PRNGKey(0)
        samples = self.run_inference(self.model, rng_key, X, Y,num_epochs)
        self.samples=samples
        
    def predict(self,X_test):
        samples = self.predictive(random.PRNGKey(1), X=X_test, Y=None)
        y=samples['Y']
        mean_prediction = jnp.mean(y, axis=0)
        percentiles = np.percentile(y, [5.0, 95.0], axis=0)
        return mean_prediction,percentiles