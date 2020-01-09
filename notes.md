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

# VAE and Z latent state
- The latent variable z describes local structure of each data point.
- Since the decoder component has to be able to reconstruct the original signal (since we're doing interpolation) then the encoder will find a representation that maximizes this information. Every value in the latent state will have some meaning that is useful in reconstructing the signal, but is unfortunately not human interpretable (like traditional time series analysis features like power).
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

# Tess

-
-
- Show most interesting object. 

# Gaia
- One billion stars in our galaxy
- Solar system - asteroids, trojan companions, kuiper belt objects and centaurs.
- Near earth objects - 1600 potentially hazardous asteroids >1km within 1 AU.
- Light bending (subtracted the sun)
- Blue photometer (330-680nm)
- Red photometer (640-1050nm)
- I resampled the data and aggregated by mean. Number of samples per object follows a distribution, even within measurement windows.
- I disabled bias
- scaling the two bands as either x/max for each or as the ratio between them, can probably lead to different interpretation.
- scaling: x/x_max where x_max is maximum for all measurements in all bands not just one. So not 2x xmaxs. Maintains relative strengths.
- scaling: x/x_max_band where we use respective max for each band. Lose relative strength but keep trends - can see when not correlated.
- scaling: log magnitude - easier to train - centred around zero but is not magnitude/distance invariant - closer objects less comparable to same objects at distance.
- interval: some intervals measured lots of objects but only had a few points, while others looked at fewer objects for longer times. I opted to filter by better measurements. Variable length!
- Added sigmoid activation at end of decoder to constrain 0-1 as we are looking at normalized max flux

# Astronomy
- Light curves. Measure brightness over time through a photometer on the spacecraft. 
- A star can still show variable brightness due to star spots (dark regions on the surface) as the star rotates on its axis.
- You need to observe for several periods (years) to confirm an orbit or other phenomena - lots of data
- Good explanation of EM spectrum - https://www.seti.org/seti-institute/project/details/seti-observations
- A Search for Analogs of KIC8462852 A Proof of Concept and First Candidates - https://iopscience.iop.org/article/10.3847/2041-8213/ab2e77
- https://www.skyandtelescope.com/astronomy-news/are-there-more-stars-like-boyajians-star/
- Lots of stellar properties we can input into models (e.g. https://exoplanetarchive.ipac.caltech.edu/cgi-bin/)
- Keck observatory is optical and infrared and has detected the presence of a supermassive blackhole SagA* (http://www.keckobservatory.org/)
- Gaia instrumentation - Blue and Red photometer for different wavelength bands - can be used as separate inputs and same procedure followed - https://en.wikipedia.org/wiki/Gaia_(spacecraft)#Scientific_instruments
- Example between BP and RP correlating and observing, and how we can encode ts info and characteristics as encoding and cluster them.
- Can do joint optical and infrared as well like Keck
- We know Tabby's star is not aliens because it absorbs specific frequencies in infrared, meaning it is not opaque.

# Light curves
- The main tool is looking for indirect evidence. 
- Measure brightness over time through a photometer on the spacecraft.
- Stars dim if an object gets in the way, which is proportional to the apparent size of that object relative to the star.
- The units we use are flux
- With low sensitivity instruments you therefore can only see objects that are massive and close to the star. This created a measurement bias in the data as all of these early planets were the size of Jupiter and in orbits similar or closer than that of Mercury. 
- You need its orbital plane to be titled toward us. People tend to think the solar system line up with the galactic plane but they unfortunately do not. Our solar system is about 60 degrees off center (which is why you can get a better view of the galactic plane in the southern hemisphere).
- To confirm a natural phenomena like a planet, you need to observe the star for long enough to see the dimming repeat. For example, someone observing the Sun would need to observe it for a year to see any corresponding dip created by the Earth. And really you would need to do this at least three times. 
- You can then take that period, and combined with properties of the Star you can determine how far away it is, and infer what the surface temperature will be, for example.

# Infrared broader EM
- What is scattered and reflected + what is absorbed is equal to what is blocked.
- Light absorbed gets emitted as heat. This heat can then be detected when looking at the object in infrared.
- Remember that infrared is not just one frequency but a range of frequencies. The particular frequencies emitted will say something about what is absorbing the light e.g. liquid water.
- So we can see opaque objects in infrared.

# Radio Astronomy
- Radio astronomy - nature doesn't like to produce narrow band signals, sources usually spill over into other frequencies. 
- https://phys.org/news/2017-09-unexpected-big-data-boom-radioastronomy.html


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

# Time series challenges (place after light curve and maybe before ODE RNN as it is the soln)
- Variable length
- Non homogenious in time
- for missing data read related work in paper

# ML
- Stress that ML is more than just classification. In reality we don't have lots of labeled data, and it can be mislabeled as this is done by humans.
- Can even use the method for spectroscopy . The non-linear way it gets encoded captures different combnations of elements and will cluster phenomena together.
- Works as a compression algorithm e.g. Gaia time series have 20k measurements and we can capture most of the information in say 100 numbers since we are training on reconstruction of the signal, and since RNNs capture non-linearities we canpnpt also capture this complexity.
- Do prepare the AE interpolation - Bernoulli distribution with p=0.25 and do element-wise product, we keep all the values for the ground truth, forces the model to learn how to understand the dynamics on the signal. 
- At a crossroads where RNNs are being used less in favor of CNNs with attention and positional encodings. This is what is used in state of the art NLP models in the form of the transformer architecture.
- Unevenly spaced, show gaps in measurements - do a heatmap like at MT
- Explain and demonstrate: Interpolation, extrapolation, concept of latent state instead of just prediction (inc. compression effect), concept of similarity, concept of search, simulate inputs then getting state and searching db with this, concept of query within db to find strange objects.
- Batching in non-homogenous time series - get the union and batch on intersection? For the sake of reducing the size of the union of time points I round to the nearest 3rd decimal. This discretization doesn't drastically reduce the number of observations. 
- Switching L1 loss for L2 I found is better at capturing small local oscillations probably since larger.
- Disabled bias

# Clustering
- Hierarcheal, spectral, density, graph based clustering
- centroid based clustering not good as it assumes similar distributions and symmetry to the clusters.
- Euclidean distance breaks in high dimensions.
- High dimensions are bizarre places and intuitive properties in 2-3 dimensions don't necessarily apply there. 
- Given the theme of this article is trying to automate away most of the process, we don't want to have to know how many clusters there are to define or what there distributions should be. We want to automatically determine this.
- Graph clusteri
- Out of all the clustering algorithms there are only a few density based approaches which allow us to not specify the number of clusters.
- http://primo.ai/index.php?title=Density-Based_Spatial_Clustering_of_Applications_with_Noise_(DBSCAN)
- https://github.com/scikit-learn-contrib/hdbscan
- Clustering, one-vs-all similarity search for most unique objects
- Clustering: OPTICS algorithm also assigns objects to no cluster if necessary. 
- If you have clusters you can then map them accross the sky or produce a power spectrum. Potentially provide insight and validation of cosmological models.
- tsne: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
- tsne: As Van der Maaten and Hinton explained: "The similarity of datapoint x j {\displaystyle x_{j}} x_{j} to datapoint x i {\displaystyle x_{i}} x_{i} is the conditional probability, p j | i {\displaystyle p_{j|i}} {\displaystyle p_{j|i}}, that x i {\displaystyle x_{i}} x_{i} would pick x j {\displaystyle x_{j}} x_{j} as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at x i {\displaystyle x_{i}} x_{i}.
- tsne: tsne has difficulty dealing with high dimensions as the ability to distinguish points by distance diminishes with increasing dimensions. 30-50 max.

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
- https://www.reddit.com/r/KIC8462852/comments/89206w/mast_tweeted_out_the_list_of_tess_target_stars/
- Say something about 1420 MHz
- Say something about shallow model but still effective - speed / sophistication trade off. 20k vecs in 16s
- Training on simulated data to then do an encoding search


<!-- decoder = nn.Sequential(
		   nn.Linear(latent_dim, latent_dim//2), 
		   nn.Tanh(),
		   nn.Linear(latent_dim//2, input_dim)) -->

<!-- self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim), nn.Sigmoid()) -->
<!-- 
self.decoder = nn.Sequential(
        nn.Linear(latent_dim, n_units),
        nn.Tanh(),
        nn.Linear(n_units, input_dim),) -->


<!-- # left ( { matrix{x_{1,1} ## x_{1,2} ## x_{2,1} ## x_{2,2}} } right )   -->
