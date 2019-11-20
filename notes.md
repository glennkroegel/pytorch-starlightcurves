# Notes on Bayesnn
If we have a network which can capture complex time series (including transients) and we have correct classification of objects, we can then begin to cluster these objects using a Bayesian NN.
The inference algorithm SVI uses a stochastic gradient estimator to take gradient steps on an objective function, which in this case is given by the ELBO (the evidence lower bound). As the name indicates, the ELBO is a lower bound to the log evidence: logp(D). As we take gradient steps that maximize the ELBO, we move our guide q(⋅) closer to the exact posterior.
The argument Trace_ELBO() constructs a version of the gradient estimator that doesn’t need access to the dependency structure of the model and guide. Since all the latent variables in our model are reparameterizable, this is the appropriate gradient estimator for our use case. (It’s also the default option.)
Elbo is KLDiv (approximate posterior) and expected log likelihood (measures model fit)
more  good text - https://pyro.ai/examples/dmm.html
https://stats.stackexchange.com/questions/309642/why-is-softmax-output-not-a-good-uncertainty-measure-for-deep-learning-models - The issue with many deep neural networks is that, although they tend to perform well for prediction, their estimated predicted probabilities produced by the output of a softmax layer can not reliably be used as the true probabilities (as a confidence for each label). In practice, they tend to be too high - neural networks are 'too confident' in their predictions. 
- Variational bayes and the local reparameterization trick: We explore an as yet unexploited opportunity for drastically improving the efficiency of stochastic gradient variational Bayes (SGVB) with global model parameters. Regular SGVB estimators rely on sampling of parameters once per minibatch of data, and have variance that is constant w.r.t. the minibatch size. The efficiency of such estimators can be drastically improved upon by translating uncertainty about global parameters into local noise that is independent across datapoints in the minibatch

# VAE
- The latent variable z describes local structure of each data point.
- The observations depend on the latent variable z in a complex, non-linear way, we expect the posterior over the latents to have a complex structure.
- Amortization is required to keep the large number of variational parameters under control.
- For a model with large observations, running the model and guide and constructing the ELBO involves evaluating log pdfs whose complexity scales badly.
- If each batch has some conditional independence we can mitigate this by subsampling. https://pyro.ai/examples/svi_part_ii.html


# ODE
- Measurements often have disontinuities 
- We can train a deep neural ODE to address this
- 
