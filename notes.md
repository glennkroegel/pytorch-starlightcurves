# Challenges
- Data can be missing or unevenly sampled.
- Observations are usually short and have transients - this makes conventional signal processing difficult.
- To get an interpretation we need the model to simply solve a task
- In the case of time series, this can be interpolation, extrapolation or classification. 

- The absense of evidence is not evidence of absense

# Notes on Bayesnn
If we have a network which can capture complex time series (including transients) and we have correct classification of objects, we can then begin to cluster these objects using a Bayesian NN.
The inference algorithm SVI uses a stochasti    c gradient estimator to take gradient steps on an objective function, which in this case is given by the ELBO (the evidence lower bound). As the name indicates, the ELBO is a lower bound to the log evidence: logp(D). As we take gradient steps that maximize the ELBO, we move our guide q(⋅) closer to the exact posterior.
The argument Trace_ELBO() constructs a version of the gradient estimator that doesn’t need access to the dependency structure of the model and guide. Since all the latent variables in our model are reparameterizable, this is the appropriate gradient estimator for our use case. (It’s also the default option.)

- Elbo is KLDiv (approximate posterior) and expected log likelihood (measures model fit). However, it is useful to anneal the KLDiv component during training. So we focus on model accuracy first and then focus on matching the distribution.

more  good text - https://pyro.ai/examples/dmm.html
https://stats.stackexchange.com/questions/309642/why-is-softmax-output-not-a-good-uncertainty-measure-for-deep-learning-models - The issue with many deep neural networks is that, although they tend to perform well for prediction, their estimated predicted probabilities produced by the output of a softmax layer can not reliably be used as the true probabilities (as a confidence for each label). In practice, they tend to be too high - neural networks are 'too confident' in their predictions. 
- Variational bayes and the local reparameterization trick: We explore an as yet unexploited opportunity for drastically improving the efficiency of stochastic gradient variational Bayes (SGVB) with global model parameters. Regular SGVB estimators rely on sampling of parameters once per minibatch of data, and have variance that is constant w.r.t. the minibatch size. The efficiency of such estimators can be drastically improved upon by translating uncertainty about global parameters into local noise that is independent across datapoints in the minibatch
- Annealing: to make training easier we make
- https://towardsdatascience.com/a-gentle-introduction-to-probabilistic-programming-languages-ba9105d9cbce
- https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e
- https://github.com/sungyubkim/MCDO/blob/master/Bayesian_CNN_with_MCDO.ipynb
- Softmax has a tendency to squeeze everything

# VAE
- The latent variable z describes local structure of each data point.
- The observations depend on the latent variable z in a complex, non-linear way, we expect the posterior over the latents to have a complex structure.
- Amortization is required to keep the large number of variational parameters under control.
- For a model with large observations, running the model and guide and constructing the ELBO involves evaluating log pdfs whose complexity scales badly.
- If each batch has some conditional independence we can mitigate this by subsampling. https://pyro.ai/examples/svi_part_ii.html


# UCR
- Show heatmap of classes and then when doing tsne show middles and fringes of clusters
- aside: norm of h (can maybe use for pword guessing)

# Uneven time
- Show plot like fig 9

# ODE
- Measurements often have disontinuities
- Latent ODEs can often reconstruct trajectories reasonably well given
a small subset of points, and provide an estimate of uncertainty over both the latent trajectories and
predicted observations.
- We can train a deep neural ODE to address this
- Variational AE framework
- A variable length sequence with one or multiple inputs is encoded into a fixed vector representation - this is how the network sematically understands the input
- Taking the union of time points does not substantially hurt the runtime of the ODE solver, as the
adaptive time stepping in ODE solvers is not sensitive to the number of time points (t 1 ...t N ) at which
the solver outputs the state. Instead, it depends on the length on the time interval [t 1 , t N ] and the
complexity of the dynamics.
- I did however round up the time points to increase training speed but this didn't result in different looking dynamics or data loss.
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
- mention dimensionality of input(
- Reconstruction/interpolation: Show an unseen sample with unevenly sampled data (with transients) and show that the reconstruction captures complexities. Mention how this non linear interpretation is then encoded in the latent state which can then be searched through and compress the signal.
- Probably should demonstrate interpolation within sampled data

# Gaia
- One billion stars in our galaxy
- Solar system - asteroids, trojan companions, kuiper belt objects and centaurs.
- Near earth objects - 1600 potentially hazardous asteroids >1km within 1 AU.
- Light bending (subtracted the sun)
- Blue photometer (330-680nm)
- Red photometer (640-1050nm)
- I resampled the data and aggregated by mean. Number of samples per object follows a distribution, even within measurement windows.
- I desabled bias

# Astronomy
- Explain flux over error and MJD
- A Search for Analogs of KIC8462852 A Proof of Concept and First Candidates - https://iopscience.iop.org/article/10.3847/2041-8213/ab2e77
- https://www.skyandtelescope.com/astronomy-news/are-there-more-stars-like-boyajians-star/
- Lots of stellar properties we can input into models (e.g. https://exoplanetarchive.ipac.caltech.edu/cgi-bin/)
- Keck observatory is optical and infrared and has detected the presence of a supermassive blackhole SagA* (http://www.keckobservatory.org/)
- Gaia instrumentation - Blue and Red photometer for different wavelength bands - can be used as separate inputs and same procedure followed - https://en.wikipedia.org/wiki/Gaia_(spacecraft)#Scientific_instruments
- Example between BP and RP correlating and observing, and how we can encode ts info and characteristics as encoding and cluster them.
- Can do joint optical and infrared as well like Keck
- We know Tabby's star is not aliens because it absorbs specific frequencies in infrared, meaning it is not opaque.

# Radio Astronomy
- Radio astronomy - nature doesn't like to produce narrow band signals, sources usually spill over into other frequencies. 


# Similarity search

- Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.
- Reference Billion Scale Similarity Search
- Curse of dimensionality
- Approximate methods are used to handle the complexity - but tend to break down for high dimensions.
- There is a requirement that the encoded data is loaded into memory, which increases proportionally to the latent dimension and number of samples. 
- Obviously for large sky surveys this can be intensive. Product quantization is a methodology used to reduce this memory footprint while maintining high speed.
- Find interesting object and plot the image frames (from tess, https://docs.lightkurve.org/quickstart.html)
- Highlighting by minimum normalized flux values we can see that they still end up in different locations on the TSNE diagram, as they have different characteristics. This is the advantage of representing time series in a non-linear way. 
- Emphasize how you don't have to have query inps that are within the dataset, you can input whatever you want.
- Gaia data is 200TB - should be possible to handle this with the above methods. We can encode lots of information into any dimension we want to save memory. More dims is more expressive but higher memory (tradeoff).
- Multiple input and output results in the same output vector, which can be added to the db. The query vector will then be the output of the neural network that took in these multiple inputs.

# ML
- Stress that ML is more than just classification. In reality we don't have lots of labeled data, and it can be mislabeled as this is done by humans.
- Can even use the method for spectroscopy. The non-linear way it gets encoded captures different combnations of elements and will cluster phenomena together.
- Works as a compression algorithm e.g. Gaia time series have 20k measurements and we can capture most of the information in say 100 numbers since we are training on reconstruction of the signal, and since RNNs capture non-linearities we can also capture this complexity.
- Do prepare the AE interpolation - Bernoulli distribution with p=0.25 and do element-wise product, we keep all the values for the ground truth, forces the model to learn how to understand the dynamics on the signal. 
- At a crossroads where RNNs are being used less in favor of CNNs with attention and positional encodings. This is what is used in state of the art NLP models in the form of the transformer architecture.
- Unevenly spaced, show gaps in measurements - do a heatmap like at MT
- Explain and demonstrate: Interpolation, extrapolation, concept of latent state instead of just prediction (inc. compression effect), concept of similarity, concept of search, simulate inputs then getting state and searching db with this, concept of query within db to find strange objects.
- Batching in non-homogenous time series - get the union and batch on intersection? For the sake of reducing the size of the union of time points I round to the nearest 3rd decimal. This discretization doesn't drastically reduce the number of observations. 
- Switching L1 loss for L2 I found is better at capturing small local oscillations probably since larger.
- Disabled bias
- Clustering, one-vs-all similarity search for most unique objects

# Datasets
- https://www.quora.com/What-are-some-astronomy-datasets-open-to-the-public
- https://datahub.io/machine-learning/spectrometer : infrared - part of IRAS low resolution Spectrometer Database Source 
- Gaia (used in search for analogs paper) - http://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/ - ts data has no ra or dec info
- Tess (basis of planet hunters) - https://archive.stsci.edu/tess/bulk_downloads.html
- National Radio Observatory - https://archive.nrao.edu/archive/advquery.jsp
- https://exoplanetarchive.ipac.caltech.edu/cgi-bin/ICETimeSeriesViewer
- Kepler - https://www.nasa.gov/kepler/education/getlightcurves - Kepler time series scripts - https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/
- Herschel Space Observatory - largest infrared telescope

# Misc
- https://twitter.com/tsboyajian/status/1181981342440906753
- Say something about 1420 MHz
