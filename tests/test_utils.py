import bilby
from aspire.samples import Samples
from aspire_bilby.utils import (
    get_prior_bounds,
    get_aspire_functions,
    get_periodic_parameters,
    sample_missing_parameters,
)
import numpy as np
import pandas as pd


def test_get_aspire_functions(bilby_likelihood, bilby_priors):
    # Intentional switching of order to test that the order of parameters is correctly handled
    parameters = ["c", "m"]
    functions = get_aspire_functions(
        bilby_likelihood, bilby_priors, parameters=parameters, use_ratio=False
    )

    theta = bilby_priors.sample(10)
    samples = Samples(
        np.array([theta[p] for p in parameters]).T,
        parameters=parameters,
    )
    logp = functions.log_prior(samples)
    samples.log_prior = logp
    logl = functions.log_likelihood(samples)

    assert np.all(np.isfinite(logp))
    assert np.all(np.isfinite(logl))


def test_get_prior_bounds(bilby_priors):
    bounds = get_prior_bounds(bilby_priors, list(bilby_priors.non_fixed_keys))
    for key in bilby_priors.non_fixed_keys:
        assert np.allclose(
            bounds[key],
            [bilby_priors[key].minimum, bilby_priors[key].maximum],
        )


def test_get_periodic_parameters(bilby_priors):
    params = get_periodic_parameters(bilby_priors)
    assert params == ["m"]


def test_sample_missing_parameters(bilby_priors, tmp_path):
    result = bilby.core.result.Result(
        outdir=tmp_path / "outdir",
        label="test",
        priors=bilby.core.prior.PriorDict({"m": bilby_priors["m"]}),
        posterior=pd.DataFrame({"m": bilby_priors["m"].sample(10)}),
        search_parameter_keys=["m"],
    )
    assert result.posterior is not None

    samples = sample_missing_parameters(result, bilby_priors)

    assert list(samples.columns) == list(bilby_priors.non_fixed_keys)
    assert samples.shape == (10, len(bilby_priors.non_fixed_keys))
    assert np.all(np.isfinite(samples.values))
