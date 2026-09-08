"""mimirax: the inference and reconstruction layer of the MIDGARD stack.

Fits, samples and diagnoses parameters of differentiable N-body forward models
through protocols, never through a specific solver.
"""

from ._typecheck import enable_runtime_typecheck

# Install the optional runtime type-checking hook before importing any mimirax
# submodules; the jaxtyping hook only instruments modules imported after it.
enable_runtime_typecheck()

from ._registry import Registry  # noqa: E402
from .diagnostics import (  # noqa: E402
    degenerate_directions,
    effective_parameters,
    fisher_information,
    has_converged,
    normalized_residuals,
    profile_likelihood,
    relative_change,
    residuals,
    rms_residual,
)
from .inference import (  # noqa: E402
    HMC,
    METHODS,
    NUTS,
    InferenceProblem,
    MadeToMeasure,
    MeanFieldVI,
    NelderMead,
    OptaxOptimizer,
    adam,
    made_to_measure,
)
from .likelihoods import LIKELIHOODS, GaussianLikelihood, chi_squared  # noqa: E402
from .observables import (  # noqa: E402
    OBSERVABLES,
    GaussianRadialBins,
    IdentityObservable,
    ProjectedPositions,
    TimeAverage,
    WeightedKernelSum,
    make_observable,
)
from .parameters import (  # noqa: E402
    REPARAMETERIZATIONS,
    AffineTransform,
    FunctionConstraint,
    IdentityTransform,
    LogTransform,
    SoftplusTransform,
    TotalMassConstraint,
    centre_of_mass,
    constrain,
    constraint_violation,
    log_abs_det_jacobian,
    net_momentum,
    num_parameters,
    quadratic_penalty,
    ravel,
    remove_centre_of_mass,
    remove_net_momentum,
    unconstrain,
)
from .priors import PRIORS, EntropyPrior, GaussianPrior, L2Regularizer  # noqa: E402
from .protocols import (  # noqa: E402
    Constraint,
    ForceModel,
    ForwardModel,
    Likelihood,
    Observable,
    Optimizer,
    Prior,
    Reparameterization,
    Sampler,
)
from .types import FitResult, SampleResult  # noqa: E402

__all__ = [
    "AffineTransform",
    "Constraint",
    "EntropyPrior",
    "FitResult",
    "ForceModel",
    "ForwardModel",
    "FunctionConstraint",
    "GaussianLikelihood",
    "GaussianPrior",
    "GaussianRadialBins",
    "HMC",
    "IdentityObservable",
    "IdentityTransform",
    "InferenceProblem",
    "L2Regularizer",
    "LIKELIHOODS",
    "Likelihood",
    "LogTransform",
    "METHODS",
    "MadeToMeasure",
    "MeanFieldVI",
    "NUTS",
    "NelderMead",
    "OBSERVABLES",
    "Observable",
    "OptaxOptimizer",
    "Optimizer",
    "PRIORS",
    "Prior",
    "ProjectedPositions",
    "REPARAMETERIZATIONS",
    "Registry",
    "Reparameterization",
    "SampleResult",
    "Sampler",
    "SoftplusTransform",
    "TimeAverage",
    "TotalMassConstraint",
    "WeightedKernelSum",
    "adam",
    "centre_of_mass",
    "chi_squared",
    "constrain",
    "constraint_violation",
    "degenerate_directions",
    "effective_parameters",
    "fisher_information",
    "has_converged",
    "log_abs_det_jacobian",
    "made_to_measure",
    "make_observable",
    "net_momentum",
    "normalized_residuals",
    "num_parameters",
    "profile_likelihood",
    "quadratic_penalty",
    "ravel",
    "relative_change",
    "remove_centre_of_mass",
    "remove_net_momentum",
    "residuals",
    "rms_residual",
    "unconstrain",
]
