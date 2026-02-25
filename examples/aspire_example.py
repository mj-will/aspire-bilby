"""Example using sequential posterior inference with ASPIRE via bilby"""

from pathlib import Path
import numpy as np
import bilby
from aspire.samples import Samples


def main():
    rng = np.random.default_rng(42)

    outdir = Path("outdir") / "aspire_example"
    outdir.mkdir(parents=True, exist_ok=True)

    # Define a simple linear model with Gaussian likelihood and uniform priors
    def model(x, m, c):
        return m * x + c

    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 1.0
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)

    priors = bilby.core.prior.PriorDict()
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")

    # Generate synthetic initial samples from a broad Gaussian away from the
    # true values
    n_initial_samples = 500
    initial_theta = rng.normal(
        loc=[1.5, 2.0],
        scale=[1.0, 1.0],
        size=(n_initial_samples, 2),
    )
    samples = Samples(initial_theta, parameters=["m", "c"])

    # Run Aspire sampler via bilby
    # This fits the flow and then performs sequential inference using the SMC
    # sampler
    # For more details on how to configure the Aspire sampler via bilby,
    # see https://aspire.readthedocs.io/projects/aspire-bilby/en/latest/usage.html
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="aspire",
        outdir=outdir,
        label="aspire_example",
        initial_samples=samples,
        injection_parameters=injection_parameters,
        n_samples=1000,
        n_final_samples=5000,
        save="hdf5",
        resume=False,
        sample_kwargs=dict(
            sampler="smc",  # Use the SMC sampler
            target_efficiency=0.8,  # Target an efficiency of 0.8 for the SMC sampler
            n_steps=20,  # Number of MCMC steps
        ),
    )

    # Plot results
    result.plot_corner()


if __name__ == "__main__":
    main()
