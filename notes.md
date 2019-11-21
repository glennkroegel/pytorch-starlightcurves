# Notes on Bayesnn
If we have a network which can capture complex time series (including transients) and we have correct classification of objects, we can then begin to cluster these objects using a Bayesian NN.
The inference algorithm SVI uses a stochasti    c gradient estimator to take gradient steps on an objective function, which in this case is given by the ELBO (the evidence lower bound). As the name indicates, the ELBO is a lower bound to the log evidence: logp(D). As we take gradient steps that maximize the ELBO, we move our guide q(⋅) closer to the exact posterior.
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


# UCR
- Show heatmap of classes and then when doing tsne show middles and fringes of clusters
- aside: norm of h (can maybe use for pword guessing)

# ODE
- Measurements often have disontinuities 
- We can train a deep neural ODE to address this
- Variational AE framework
- A variable length sequence with one or multiple inputs is encoded into a fixed vector representation - this is how the network sematically understands the input
- This representation can be understood as a point in space, with closer points being more similar (lat/long) - but in a higher dimensional - tsne to visualize
- At the same time, we train a decoder to output a fixed dimension time series which will fill in the spaces.
- This latent variable framework comes with several benefits: First, it explicitly decouples the dynamics
    of the system (ODE), the likelihood of observations, and the recognition model, allowing each to be
    examined or specified on its own. Second, the posterior distribution over latent states provides an
    explicit measure of uncertainty, which is not available in standard RNNs and ODE-RNNs. Finally, it
    becomes easier to answer non-standard queries, such as making predictions backwards in time, or
    conditioning on a subset of observations.
- Discuss how poisson processes in the latent space is beneficial and what insights it gives - the fact that a measurement was made at a particular time can be informative about a system - this might not be the case about a sky survey where thousands of measurements are done at once, but is for medical data i.e. someone becoming sick and taking a measurement.
- When to use ode-rnn: standard rnns ignore the time gaps between points, with few missing values, or when time intervals between points are short. In ode-rnns the dynamics between time points is learned.
- interpolation and extrapolation possible
- to do extrapolation train on the first half and decode the second half. (so z can be a repr of whole signal or first half, rnn-vae is full signal interpretation)
- mention dimensionality of input