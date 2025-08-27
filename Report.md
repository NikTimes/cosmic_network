#### Background

The purpose of this paper is not only to showcase a surrogate model for all densities in the $\Lambda$CDM model but also to shine a light on machine learning applied to Computational Physics. That is why, in the background section of this research, we will not only discuss the physics and computational challenges of deriving the angular power spectrum, but also provide a mathematical treatment of the learning algorithms underlying neural networks, with particular emphasis on backpropagation. We organize the section in the following way:

- Angular Power Spectrum - brief review of physical origin and relevance in cosmology
- CAMB and CLASS - Overview of Boltzmann solvers
- Backpropagation - derivation of training algorithm
- Training Hyperparameters - key learning parameters 
- PCA - Role of dimensionality reduction 

##### Angular Power Spectrum 

The goal of this paper is to accurately predict the CMB angular power spectrum and do so efficiently. To tackle and properly highlight the issues of current methods to achieve such a task we must first have a comprehensive idea of what is the CMB angular power spectrum and why it is central to cosmology in the first place. 

###### What is the CMB?

The CMB is the earliest possible electromagnetic image that we can currently observe. Prior to its release, the universe was a hot, dense plasma in which all components of the standard model were in thermal equilibrium with each other, including matter and radiation. 

In this tightly coupled state, photons interacted continuously with the surrounding plasma, primarily through Thomson scattering off free electrons. As a result, radiation could not propagate freely which made direct observation impossible. 

To model the expanding universe during this epoch we use the **flat Friedmann–Lemaître–Robertson–Walker (FLRW) metric**

$$ds^2 = -c^2dt^2 + a^2(t)\left[dr^2 + r^2(d\theta^2 + \sin^2\theta d\phi^2) \right] \quad (1) $$

Where $a(t)$ is the **scale factor** driving the expansion of the universe. 

As $a(t)$ grows the universe cools according to $T \propto 1/a(t)$. This cooling not only suppresses the average energy of photons, but also enables the formation of stable neutral hydrogen atoms. As a result, processes such as Thomson scattering ceased to be efficient, allowing photons to decouple from matter. These photons have since propagated freely across the universe, forming the Cosmic Microwave Background.

###### Angular Power Spectrum

If we point a radio antenna at the sky and measure the CMB, we find that while the observed photon spectrum follows a nearly perfect blackbody with constant average temperature, a detailed analysis using the Planck distribution reveals tiny anisotropies. 

We can define the CMB temperature anisotropies as a function on the celestial sphere: 
$$\Theta(\hat n) = \frac{\Delta T(\hat n)}{T}$$

Where $\hat n$ denotes a direction on the sky. Since the sky is a two-sphere, we can express this function in spherical harmonics as:
$$\Theta(\hat n) = \sum_{\ell=0}^{\infty} \sum_{m = -\ell}^\ell a_{\ell m}Y_{\ell m}(\hat n)$$
Because the CMB is well described as a nearly Gaussian random field, all of its statistical information is contained in its **two-point correlation** function:
$$C(\theta) = \langle\Delta T(\hat n)\Delta T(\hat n')\rangle$$

Statistical Isotropy then leads to: 

$$\langle a_{\ell m} a_{\ell' m'}^*\rangle = C_{\ell} \delta_{\ell \ell'} \delta_{mm'}$$

Therefore we have that the set of coefficients driving the spherical harmonic decomposition of $\Theta(\hat n)$ is completely described by the set of coefficients {$C_\ell$}. These make up the **Angular power spectrum**. 

Currently our most precise measurement of the angular power spectrum was carried out by Planck 2018. Any viable cosmological theory must be able to reproduce the main features of that spectrum within uncertainty.  

_The 6-parameter ΛCDM model continues to provide an excellent fit to the cosmic microwave background data at high and
low redshift, describing the cosmological information in over a billion map pixels with just six parameters. With 18 peaks in the temperature
and polarization angular power spectra constrained well, Planck measures five of the six parameters to better than 1% (simultaneously), with the
best-determined parameter (θ∗) now known to 0.03%. We describe the multi-component sky as seen by Planck, the success of the ΛCDM model,
and the connection to lower-redshift probes of structure formation. We also give a comprehensive summary of the major changes introduced in
this 2018 release_ [Abstract](https://www.aanda.org/articles/aa/pdf/2020/09/aa33880-18.pdf)

_We find good consistency with the standard spatially-flat 6-parameter_ [Abstract](https://arxiv.org/abs/1807.06209?utm_source=chatgpt.com)

Planck finds that the six-parameter $\Lambda CDM$ model provides an excellent fit to the angular power spectrum. $\Lambda CDM$ is characterized by the fractional energy densities of the main components of the Universe, - Baryons ($b$), dark matter $\text{cdm}$, photons ($\gamma$), relativistic neutrinos ($\text{ur}$), dark energy ($\Lambda$) and curvature ($k$) - denoted by $\Omega_i$. These the follow the friedmann constraint.

$$\Omega_{\Lambda} + \Omega_k + \Omega_\text{cdm} + \Omega_b+ \Omega_\text{ur} + \Omega_\gamma = 1$$

 While the Planck analysis is performed in terms of the following independent parameters $(\Omega_b h^2,\, \Omega_c h^2,\, 100\,\theta_s,\, \tau,\, n_s,\, \ln(10^{10}A_s))$ these uniquely determine the values of the density fractions above. Varying these $\Omega_i$ allows us to study how each component of the cosmic fabric affects the CMB power spectrum. 

#### Boltzmann solvers

As mentioned above Planck 2018 provides precise constraints that any cosmology should follow. To connect these observations with theory we must carry out a perturbed general relativistic treatment of how fluctuations in the early universe evolve with its expansion. As emphasized in Weinberg {weinberg2008}, the observed anisotropies in the CMB arise from several key physical effects:

- {Intrinsic temperature fluctuations} in the photon–baryon plasma at last scattering (redshift $z \simeq 1090$).
- {Doppler effect} from velocity perturbations of the plasma at recombination.
- {Sachs–Wolfe effect}: gravitational redshift or blueshift due to potential fluctuations at last scattering.
- {Integrated Sachs–Wolfe effect}: additional redshifts or blueshifts caused by time-varying gravitational potentials along the photon’s path from last scattering to today.

After a full treatment of these effects we arrive to  the Einstein-boltzmann equations, which must be solved for all species (photons, baryons, dark matter, neutrinos, dark energy) in order to accurately predict the CMB angular power spectrum. From linear theory we have that $C_\ell^{XY}$ ($X,Y \in \{T,E,B\}$) are given by: 

$$C_\ell^{XY} = 4\pi \int_0^\infty \frac{dk}{k}\,
\mathcal{P}_{\mathcal{R}}(k)\, \Delta_\ell^X(k)\, \Delta_\ell^Y(k),
$$

where $\mathcal{P}_{\mathcal{R}}(k)$ is the primordial curvature power spectrum and 
$\Delta_\ell^X(k)$ are radiation transfer functions. These transfer functions are computed through the line-of-sight integral (Seljak & Zaldarriaga 1996): 

$$\Delta_\ell^X(k) = \int_0^{\eta_0} d\eta \;
S_X(k,\eta)\, j_\ell\!\big(k[\eta_0-\eta]\big),$$

with $j_\ell$ spherical Bessel functions and $S_X(k,\eta)$ the source functions encoding Sachs–Wolfe, Doppler, polarization, integrated Sachs–Wolfe, reionization, and lensing effects. Computing these source functions requires solving the full Einstein-boltzmann Hierarchy. For photons that looks like the following:

$$\dot\Theta_\ell = \frac{k}{2\ell+1}\big[\ell\,\Theta_{\ell-1} - (\ell+1)\,\Theta_{\ell+1}\big]
- \dot\tau\left(\Theta_\ell - \delta_{\ell 0}\Theta_0^{(\text{src})} - \delta_{\ell 1} v_b - \cdots\right),$$

with $\dot\tau$ the Thomson scattering rate and $v_b$ the baryon velocity, coupled in turn to baryons, CDM, neutrinos, and metric perturbations. From the structure of the Einstein–Boltzmann hierarchy it is evident that computing the CMB angular power spectrum is highly demanding: each multipole $\Theta_\ell$ is coupled to its neighbors $\Theta_{\ell-1}$ and $\Theta_{\ell+1}$, producing a large system of coupled differential equations. To obtain accurate predictions up to small angular scales, the hierarchy must be evolved up to $\ell_{\max} \sim 3000$, corresponding to thousands of coupled equations that must be solved for every Fourier mode $k$. 

To tackle this challenge efficient **Boltzmann solvers** have been developed. Most notably these include \texttt{CAMB} (Lewis, Challinor \& Lasenby 2000) and \texttt{CLASS} (Lesgourgues 2011). However, although these solvers are highly efficient and provide accurate theoretical predictions for comparison with Planck, the scale of the computation still means that each run can take up to seconds. While this compute time is fast enough for a lot of applications, it is not suitable for fast **Real-Time applications**, thus motivating the creation of a **surrogate model**. 

### Surrogate models 

[surrogate_models_in_cosmology](https://iopscience.iop.org/article/10.1088/1361-6633/acd2ea/meta)
[gravitational_waves](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.4.031006)
[cosmopower](https://academic.oup.com/mnras/article/511/2/1771/6505144)

As the fields of Data Analysis and Machine Learning evolve their impact extends across all areas of science, cosmology is being no exception. A prominent example of this influence is the adoption _surrogate models_. 

[31 European Symposium on Computer Aided Process Engineering 2021, Computer Aided Chemical Engineering Bianca Williams, Selen Cremaschi](https://www.sciencedirect.com/topics/materials-science/surrogate-modeling#:~:text=Surrogate%20models%20are%20simplified%20approximations,be%20appropriate%20for%20an%20application.)

"Surrogate models are simplified approximations of more complex, higher-order models. They are used to map input data to outputs when the actual relationship between the two is unknown or computationally expensive to evaluate. Surrogate models are of particular interest where expensive simulations are used \[...]". 

In this project, we focus on the use of neural networks (NN) as effective surrogates for predicting the temperature power spectrum of the CMB. However, much of the existing literature employing such techniques assumes that the reader is already familiar with their underlying principles. As a result, neural networks are often presented as opaque ‘black boxes,’ which can leave many researchers either unconvinced by their reliability or uninspired to engage with their use. In the next few sections of this paper we focus on building up these principles and answering why they make highly effective surrogates. 

##### The Mathematics Behind Neural Networks
A neural network is a parameterized function

$$f_\theta : A \to B$$

where $A$ denotes the space of inputs, $B$ the space of outputs, and $\theta$ the collection of learnable parameters. The idea is that neural networks approximate complex, often nonlinear, mappings by composing simple transformations.

We begin with the one–dimensional case $f_\theta:\mathbb{R}\to\mathbb{R}$. Let $x\in\mathbb{R}$ be the input and set $h_0 := x$. For layers $j=1,\dots,L$ we alternate a linear (affine) map and a nonlinearity:

$$
\text{(Two-step)}\quad
z_j = w_j h_{j-1} + b_j,\qquad h_j = \sigma_j(z_j),
$$

or equivalently

$$
\text{(One-line)}\quad
h_j = \sigma_j\!\big(w_j h_{j-1} + b_j\big),\qquad h_0 = x,
$$

with $w_j,b_j\in\mathbb{R}$. The network output is $f_\theta(x):=h_L$. In composed form, 

$$
f_\theta(x)=\sigma_L\!\Big(w_L\,\sigma_{L-1}\big(\cdots\,\sigma_1(w_1 x + b_1)\,\cdots\big)+b_L\Big).
$$

The parameters $\theta=\{(w_j,b_j)\}_{j=1}^L$ are initialized (typically randomly) and then trained on a dataset

$$
\mathcal{D}=\{(x^{(i)},y^{(i)})\}_{i=1}^N \subset \mathbb{R}\times\mathbb{R},
$$  
Where, $x^{(i)}$ denotes the $i$-th input sample, while $y^{(i)}$ is its corresponding output according to the mapping we aim to approximate. Thus the collection $\{x^{(i)}\}_{i=1}^N$ forms the set of inputs, and $\{y^{(i)}\}_{i=1}^N$forms the set of outputs otherwise known as **labels**.  

Now let $\mathcal{L}: \mathbb{R} \times \mathbb{R} \to [0, \infty)$ be an arbitrary **loss function** which measures the discrepancy between a predicted output $\hat y=f_\theta(x)$ and the true output $y$ (for example $\mathcal{L}(\hat y,y)=\tfrac12(\hat y-y)^2$). The average discrepancy over the training set is called **empirical risk**.

$$J(\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \mathcal{L}\!\big(f_\theta(x^{(i)}),\,y^{(i)}\big),
  \qquad \theta=\{(w_j,b_j)\}_{j=1}^L$$

The objective during training is to find parameters that minimize **$J(\theta)$**. To find such set of parameters the network undergoes the **gradient descent.** Note that the gradient of the empirical risk with respect to the parameters is defined as: 

$$\nabla_\theta J(\theta)  \;=\;
  \left(
    \frac{\partial J}{\partial w_1},\frac{\partial J}{\partial b_1},
    \dots,
    \frac{\partial J}{\partial w_L},\frac{\partial J}{\partial b_L}
  \right).$$

and points in the direction of the steepest local **increase** of $J(\theta)$. To minimize $J(\theta)$ we input the dataset through the network and use the labels to calculate the empirical risk we adjust the parameters in the opposite direction of the gradient: 

$$\begin{align}
w_j &\;\rightarrow\; w_j - \eta\,\frac{\partial J}{\partial w_j}, \quad j=1, \dots,L, 
\\
b_j &\;\rightarrow\; b_j - \eta\,\frac{\partial J}{\partial b_j}, \quad j=1, \dots,L,
\\
\end{align}$$

In practice we compute $\nabla_\theta J$ via **backpropagation**. Each step performs a forward pass to form predictions, computes the empirical risk against labels, backpropagates gradients, and updates parameters; an **epoch** is one full pass over the dataset, and training typically runs for multiple epochs.

**From scalars to vectors.** The general case replaces scalars with vectors and matrices. For a layer of width $d_j$, let $h_{j-1}\in\mathbb{R}^{d_{j-1}}$, $W_j\in\mathbb{R}^{d_j\times d_{j-1}}$, and $b_j\in\mathbb{R}^{d_j}$. The forward map is

$$h_j = \sigma_j(W_j h_{j-1} + b_j), \qquad j = 1, \dots , L$$

Where $\sigma_j$ acts elementwise. Thus $f_{\theta}: \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$ with $\theta = \{ W_j, b_j\}_{j = 1}^L$. Widths $(d_1,\dots,d_{L-1})$ and depth $L$ control capacity; training, losses, and gradient-based optimization proceed exactly as in the scalar case. 
 
##### Training parameters relevant to this project: 

**Depth and Width:** How many layers and how many neurons per layer in the model. More complexity will capture better dynamics but comes with the risk of overfitting.  

**Activation Functions:** Non-linear functions applied after each linear layer, they let the model learn curved relationships instead of just straight lines. Common choices include: 

- $\mathrm{ReLU}(x)=\max(0,x)$. 
- $\tanh$: Outputs in $(-1,1)$

**Batch Size:** How many examples you process before taking one optimization step.

**Learning Rate:** In the update step the scalar $\eta > 0$ controls the step size in parameter space. Too large → divergence; too small → slow, possibly stuck. Common practice: start with $\eta\in[10^{-4},10^{-2}]$


**Learning Rate Scheduler:** Schedules change the value of the learning rate across epochs to avoid divergence or plateuing. Some schedulers might change according to some function for instance cosine or exponential. 

 
##### PCA 

Finally the last relevant component of our project is the use of **Principal Component Analysis** 

### Methodology 


In order to train our surrogate neural network we can divide the methodology in three steps:

- Creating the dataset
- Training 
- Testing 

#### Creating the dataset 

We begin by specifying the parameter ranges for the density components of interest:

$$\begin{align} 
\Omega_b h^2 &\in [0.001, 0.40] \\ 
\Omega_{\text{cdm}} h^2 &\in [0.001, 0.40] \\
\Omega_{\gamma} h^2 &\in [2.30\cdot 10^{-5}, 2.60\cdot 10^{-5}] \\
\Omega_{\text{ur}} h^2 &\in [1.30\cdot 10^{-5}, 2.00\cdot 10^{-5}]
\end{align}$$

Once the ranges are defined, we sample density quadruples $(\Omega_b, \Omega_{\text{cdm}}, \Omega_{\gamma}, \Omega_{\text{ur}})$ within these intervals and use them as inputs to CAMB in order to compute the corresponding CMB power spectra. 

Note, however, that the parameter quadruples must be sampled **uniformly across all density ranges**, which requires a sampling scheme that ensures proper coverage of the parameter space. A standard choice for this is **Latin Hypercube Sampling (LHS)**. Unlike simple random sampling, LHS divides each parameter dimension into equal-probability intervals and draws exactly one sample from each interval, thereby guaranteeing a more uniform and space-filling coverage. 

Since for the scope of this project we restrict ourselves to a flat universe ($k = 0$) the following constraint must be satisfied: 

$$\Omega_b h^2 + \Omega_{\text{cdm}} h^2 + \Omega_{\gamma} h^2 +
\Omega_{\text{ur}} h^2  + \Omega_{\Lambda}h^2 = h^2$$

In other words we may calculate the dark energy contribution $\Omega_{\Lambda}h^2$ explicitly using:

$$\Omega_{\Lambda}h^2 = h^2 -\Omega_b h^2 - \Omega_{\text{cdm}} h^2 - \Omega_{\gamma} h^2 - \Omega_{\text{ur}} h^2 $$

A parameter set is considered physical only if

$$1 \geq \Omega_{\Lambda} \geq0$$

If this is not satisfied, we discard the quadruple and resample. Otherwise we store the quadruple together with the corresponding CMB power spectra in an **HDF5** file for training. This file format provides both efficient storage and fast access, making it particularly well-suited for machine learning applications.

(Add paragraph explaining PCA)

(Add final paragraph explaining findings and how many samples were discarded and how many you ended up needing for training)

#### Training 

(Will depend on Background)

#### Testing

Once training is complete, the neural network is evaluated on a **separate test set** consisting of parameter samples and corresponding spectra that the model has never encountered during training. This requires generating an additional dataset with CAMB, independent of both the training and validation sets, since reusing those would introduce bias and overestimate the model’s performance.

To quantify accuracy, we compute both standard machine learning metrics such as the mean squared error (MSE) and the coefficient of determination ($R^2$), as well as physics-specific measures including the fractional error ($\epsilon_l$) in the reconstructed $C_ℓ$​ spectra and the accuracy of acoustic peak locations. Which we define as follows: 

- **Mean Fractional Error:** $$|\epsilon_{\ell}| = \frac{1}{\ell_{\text{max}} - \ell_{\text{min}}}\sum_{\ell = \ell_{\text{min}}}^{\ell_{\text{max}}}\left| \frac{C_{\ell}^{\text{pred}} - C^{\text{CAMB}}_{\ell}}{C^{\text{CAMB}}_{\ell}} \right|$$
- **Max Fractional Error:** $$\max_\ell​∣ϵ_ℓ​∣$$



### Radiation 

[neutrino and radiation density](https://physics.stackexchange.com/questions/94181/where-is-radiation-density-in-the-planck-2013-results?utm_source=chatgpt.com)
[planck cosmological parameters](https://arxiv.org/pdf/1303.5076)

$T_0 = 2.7255 + 0.0006$K according to Planck data we can keep $T_0 = 2.7255$K moreover we have that neutrino density is related to photon density by:

$$\rho_{\nu} = N_{\text{eff}} \frac{7}{8} \left(\frac{4}{11} \right)^{4/3} \rho_{\gamma}$$


where $N_{\text{eff}} = 3.046$ and $\rho_{\gamma}$ can be derived from the planck law: 

$$\rho_\gamma c^2 = \int_0^\infty h\nu\,n(\nu)\,d\nu 
= a_B T_0^4,$$

Where $a_B = \frac{8\pi^5 k_B^4}{15 h^3 c^3} = 7.56577 \times 10^{-16}\ \mathrm{J\,m^{-3}\,K^{-4}}$ is the radiation energy constant. Plugging $T_0$ we obtain that: $$\begin{align*} \rho_\gamma = \frac{a_B T_0^4}{c^2} 
&= 4.645 \times 10^{-31}\ \mathrm{kg\,m^{-3}} \\ \implies
\rho_{\nu}  &= 3.213 \times 10^{-31}  \mathrm{kg\,m^{-3}}\end{align*}$$
Divide this by the critical density [value of critical density](https://en.wikipedia.org/wiki/Friedmann_equations#Critical_density) $\rho_{\text{crit}}=  \frac{3H_0^2}{8 \pi G} = 1.88 \times 10^{-26} h^2  \mathrm{kg\,m^{-3}}$:  

$$\begin{align*}\Omega_{\gamma} h^2 = 2.471 \times 10^{-5} \\ \Omega_{\text{ur}}h^2 = 1.709 \times 10^{-5} \end{align*}$$



| Parameter                            | Value                  |
| ------------------------------------ | ---------------------- |
| $\Omega_b h^2$                       | $0.02237 \pm 0.00015$  |
| $\Omega_c h^2$                       | $0.1200 \pm 0.0012$    |
| $100\theta_{\rm MC}$                 | $1.04092 \pm 0.00031$  |
| $\tau$                               | $0.0544 \pm 0.0073$    |
| $\ln(10^{10} A_s)$                   | $3.044 \pm 0.014$      |
| $n_s$                                | $0.9649 \pm 0.0042$    |
| $H_0 \,[{\rm km\,s^{-1}\,Mpc^{-1}}]$ | $67.36 \pm 0.54$       |
| $\Omega_\gamma h^2$                  | $2.471 \times 10^{-5}$ |
| $\Omega_\nu h^2$                     | $1.709 \times 10^{-5}$ |

