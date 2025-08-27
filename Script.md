
### What is the CMB 

In very short words, the CMB is the earliest possible snapshot that we have of the universe **using electromagnetic waves**. Why? To answer this question we need to go back to the universe right after the big bang. 

At this point in time the universe can be thought of as a hugely dense and hot thermal bath where all components of the universe are at thermal equilibrium from each other and radiation dominates. At this point the energy is so high that all particles in the standard model are being formed. 

As the universe expands and cools, certain high-energy processes become energetically inaccessible, the particles in the thermal bath no longer have sufficient energy to sustain these interactions. For example, when the temperature drops below the mass-energy threshold of tau leptons, tau production stops, causing them to fall out of equilibrium and effectively decouple from the thermal bath.

This keeps going until eventually neutral atoms begin to form and the photons in the bath (redshifted by the expansion of the universe) no longer have the energy to ionize them or scatter off free electrons marking the beginning of Recombination and the period of last Scattering. 

Photons are now decoupling from matter and begin to travel freely. Permeating the universe and essentially making it transparent. These photons ARE what make up the CMB. 

By doing a statistical analysis of this period we can come up with this equation  which tracks the ratio of free $X_e$ electrons in the thermal bath.:

$$\frac{1 - X_e}{X_e^2} = \frac{1}{n_b} \left( \frac{2\pi m_e k_B T}{h^2} \right)^{3/2} e^{E_i / k_B T}$$

Solving this equation we see that $X_e$ drops sharply at $T = 3000K$, indicating that the formation of neutral hydrogen and thus the time of last scattering happened around $380.000$ years after the big bang.

### What do we learn from the CMB 

If we point a radio antenna at the sky and measure the incoming photons, we observe slightly different temperatures coming from different directions by about a factor of $10^{-5}$. These tiny variations in temperature known as **temperature anisotropies**. Anisotropies contain valuable information about the distribution of matter at the time of last scattering. 

The hotter spots correspond to denser regions where gravity compressed the thermal bath and increased the photons energy. The colder spots, on the other hand, represent underdense regions where the gravitational potential was shallower. 

These anisotropies in the CMB can be statistically described in fact, since the CMB is observed over a spherical sky, the temperature field $\Delta T(\hat{n})$  can expanded in spherical harmonics:

$$\Delta T(\hat{n}) = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell} a_{\ell m} Y_{\ell m}(\hat{n})$$

Assuming statistical isotropy and Gaussianity, the CMB fluctuations are fully 
described by the ensemble average:

$$\langle a_{\ell m} a^*_{\ell' m'} \rangle = C_\ell \delta_{\ell \ell'} \delta_{m m'}$$

Where $C_\ell$ is the **angular power spectrum** it quantifies the variance of temperature fluctuations at angular scale $\theta \sim 180^\circ / \ell$. In cosmology, if you are able to come up with a model that fits the power spectrum, you are probably on the right track for a valid theory of the history of the universe. Why don't we give it a go?

### Our Universe

Heuristically, it makes sense to think that temperature at the time of last scattering has a big effect on what the CMB looks like today. Realistically there is a whole bunch of other effects that matter all of which require a general relativistic approach but for now lets focus on temperature fluctuations. 

1. Intrinsic temperature fluctuations in the electron–nucleon–photon
plasma at the time of last scattering,

2. The Doppler effect due to velocity fluctuations in the plasma at last
scattering.

3. The gravitational redshift or blueshift due to fluctuations in the
gravitational potential at last scattering. This is known as the Sachs–
Wolfe effect

4. Integrated Sachs–Wolfe effect

The origin of these fluctuations stem once again from the thermal bath. While matter creates gravitational wells that pull in and compress the thermal bath, radiation pressure pushes back, resisting compression. This tug-of-war between gravity and radiation pressure generates **pressure (acoustic) waves** in the photon–baryon fluid. At the time of last scattering, these oscillations "freeze," seeding the CMB anisotropies we observe today.

Clearly, whether or not the universe is dominated by radiation or matter has a a big impact on the way that these pressure waves behave. As I said before Radiation dominated the early universe so one might assume that the universe was radiation dominated at last scattering, however radiation density dilutes faster with expansion than matter density, so at some point, they become equal:

$$\rho_{r}(a)\propto a^{-4}, \quad \rho_{m}(a)\propto a^{-3} \ \implies \rho_r = \rho_m$$

This moment is called **matter–radiation equality**.

- **If equality happens before last scattering**:  
  
  The universe becomes **matter-dominated**: gravity dominates, allowing strong compressions in the plasma.

- **If equality happens after last scattering**: radiation pressure still dominates, suppressing these compressions and damping the anisotropies.

To study this phenomena we can use the Friedmann equation to get an estimate of when the matter-radiation equality happen: 
$$H^2 = \left(\frac{\dot a}{a}\right)^2 = \frac{8 \pi G}{3}\rho(t)$$
Where $a$ is the scalar factor determining how the universe expands and $\rho$ is the total density determined by the sum of the densities of the components of the universe, baryonic matter and photons in our case.  $$H^2 = \left(\frac{\dot a}{a}\right)^2 = \frac{8 \pi G}{3}\left(\frac{\rho_{b, 0}}{a^3} +  \frac{\rho_{r, 0}}{a^4}\right)$$In cosmology we prefer to write this formula in dimensionless units the following way:$$\left(\frac{\dot a}{a}\right) = H_0 \sqrt{\frac{\Omega_{b}}{a^3}  + \frac{\Omega{r}}{a^4}} $$Where 

- $\Omega_i = \frac{\rho_{b, 0}}{\rho_{\text{crit}}}$
- $\rho_{crit} = \frac{3H_0}{8 \pi G}$

When we solve this equation using the Planck values for $\Omega_b$ and $\Omega_r$ we will obtain that the matter-radiation equality happens after the time of last scattering. We can use a **Boltzmann Solver** to do a full treatment of the effects that go in the CMB and plot the power spectrum we obtain that the power spectrum obtained is not at all similar to the one we expect. 

In order to solve this problem, let's examine the case where the matter radiation equality to happens before the time of last scattering. To do this we are going to add another mass term into our equation and run the same programs

$$\left(\frac{\dot a}{a}\right) = H_0 \sqrt{\frac{\Omega_{b}}{a^3} + \frac{\Omega_{\text{cdm}}}{a^3}  + \frac{\Omega{r}}{a^4}}$$

This time we see that our simulation matches the Planck power spectrum much better. However, this is still incomplete. Observations by **Hubble** and later by **Saul Perlmutter, Brian P. Schmidt, and Adam G. Riess et al.** showed that the universe's expansion is not only ongoing ($\dot{a} > 0$) but also **accelerating** ($\ddot{a} > 0$).

If we evolve the Friedmann equation with only **matter** and **radiation**, we find that while $\dot{a} > 0$, the expansion is slowing down ($\ddot{a} < 0$), because both components dilute over time. But observations of distant supernovae reveal that this deceleration eventually **reverses into acceleration**.

To explain this, we introduce a new component: **Dark Energy ($Ω_Λ$)**. Unlike matter and radiation, dark energy’s density remains constant as the universe expands. As the other components dilute, $Ω_Λ$ becomes dominant, driving the universe into a phase of accelerated expansion.

$$\left(\frac{\dot a}{a}\right) = H_0 \sqrt{\frac{\Omega_{b}}{a^3} + \frac{\Omega_{\text{cdm}}}{a^3}  + \frac{\Omega{r}}{a^4} + \Omega_{\Lambda}}$$

This concludes our analysis for the CMB in this analysis 

- we learnt that $\Lambda CDM$ depends on $\Omega_s$ 
- We learnt how omegas affect the CMB 
- We learnt CMB can be computed solely from power spectrum
- We can simulate the Power spectrum using a Boltzmann Solver

some lambda stuff

## Project

Going back to Venus's project we have now found 4 constants that we can vary in order to change the outcome of the CMB and moreover we have also learnt to appropriately simulate it using a Boltzmann solver. Python library called (CAMB)

Has this been done before? and the answer is that Yes! In fact the very planck consortium providing us with the values of lambda made a simulator that does exactly what I described. 

However this simulator comes with its very own problems. The most obvious one being that it does not include a slider for radiation only baryonic matter, cold dark matter and energy. 

Another big aspect to mention is that all of the images shown in the simulator are precomputed. In other words this is not a real time simulation. Whilst this is a perfectly fine approach to creating a tool like this. It comes with the drawback that it limits your slider step size to whatever you have precomputed your power spectra to. In this case the slider only allows you to modify the omegas in steps of 0.025 units. From here we can outline the following improvements and build up our objectives for this project.

- Planck-like CMB simulator:
- Allowing radiation.
- Allow for Real time computation of power spectra
- Continuous slider
### Why use a Neural Networks

The first thing that might come to mind in order to fulfill these objectives, is to directly use CAMB to compute the power spectra in real time. However if you try to do this you will never be able to reach those real time computations that we desire. 

That is because CAMB does a full treatment of the linear Boltzmann equations for the evolution of cosmological perturbations, accounting for all the relevant physical processes that have an effect in the CMB. This computation involves solving a large set of coupled differential equations with high precision, which is very computationally intensive and can take several seconds to minutes per run

In order to fulfill our objectives we must find a way to optimize CAMB. Here is where we introduce the wonders of machine learning. Particularly the wonders of **Neural Networks.**

A Neural Network can be thought of as a sequence of matrix operations that progressively transform an input vector into a desired output. Specifically, in our case we represent $\Omega_b, \Omega_{cdm}, \Omega_{\gamma}, \Omega_{ur}$ as a vector. Then this vector is successively multiplied by a weight matrix which shapes the input into a vector that more closely resembles the output. After each multiplication, a non-linear activation function is applied, allowing the network to model complex, non-linear relationships between inputs and outputs. This is repeated through a predetermined number of hidden layers until we get an output. In our case the Power Spectrum. 

These weight metrices are initially assigned random values which means that at first our output will be a random mess of noise. In order to train our network to correctly predict the power spectrum. We precompute pairs of Input Omegas with their correspondent Power spectrum using CAMB. We then pass our inputs through the network to get a prediction and compare it with the real power spectrum. We calculate the MSE $(\text{out} - \text{targ})^2$ and use this error to slightly change the weights of the network. We repeat this process until the error converges. 

Once the training phase is completed, our neural network becomes capable of reproducing the power spectrum for the cosmological parameter values it has seen during training. More importantly, it can also predict the power spectrum for new input parameter combinations it has never encountered before.

In the end, we obtain a model that effectively emulates CAMB, capable of reproducing the power spectrum for any physically reasonable combination of Omegas. However, unlike CAMB, which requires computationally expensive integrations of the Boltzmann equations, the neural network accomplishes this task through a series of simple matrix multiplications. 

This dramatically reduces the computation time from several seconds or minutes per model to mere milliseconds, enabling real-time exploration the power spectrum for a continuous range of Omegas. The only drawback being that you need a large dataset in order to train such a model. 

However, making that dataset is not hard, since we can just define a range of values for all our omegas have a random sampler select from these values, plug it into CAMB and obtain our desired spectrum for said sample of omegas. Repeat this process 80k times and you have a dataset. 






$$\int_{-\infty}^{\infty} \left||x|e^{-\frac{m\alpha|x|}{\hbar^2}}\right|^2 = \int_{-\infty}^{\infty} |x|^2e^{-\frac{2m\alpha|x|}{\hbar^2}} = \int_{-\infty}^{\infty} x^2e^{-\frac{2m\alpha |x|}{\hbar^2}}$$

$$\int_{-\infty}^{\infty} x^2e^{-a|x|}dx = [ -\frac{x^2}{a}e^{-a|x|}]_{-\infty}^{\infty} + \int_{-\infty}^{\infty} \frac{2x}{a}e^{-a|x|}dx $$

$$\frac{n!}{a^{n+1}} = \int_{0}^{\infty}x^n e^{-ax}$$

$$\int_{-\infty}^{\infty}x^2 e^{-a|x|} = 2\times\int_{0}^{\infty}x^2 e^{-a|x|} = \frac{2\times2!}{a^{3}} = \frac{4}{a^3}$$
