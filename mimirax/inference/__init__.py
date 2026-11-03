"""Inference methods -- optimizers and samplers -- and their registry.

Every entry satisfies :class:`~mimirax.protocols.Optimizer` or
:class:`~mimirax.protocols.Sampler`. :class:`OptaxOptimizer` and
:class:`MadeToMeasure` are implemented, as is :class:`DataResampling` for
uncertainties on made-to-measure weights, and :class:`AugmentedLagrangian` for
equality-constrained problems (multiple shooting; it takes the problem, not a
bare objective, so it is not registered as an ``Optimizer``); :class:`HMC`,
:class:`NUTS`, :class:`MeanFieldVI` and :class:`NelderMead` are stubs that fix
names and signatures. Register new methods with ``METHODS.register(name, cls)``.
"""

from mimirax._registry import Registry
from mimirax.inference.augmented_lagrangian import (
    AugmentedLagrangian,
    AugmentedLagrangianResult,
    augmented_lagrangian_value,
    defect_violation,
)
from mimirax.inference.derivative_free import NelderMead
from mimirax.inference.hmc import HMC, NUTS
from mimirax.inference.m2m import MadeToMeasure, made_to_measure
from mimirax.inference.objective import InferenceProblem, ProblemEvaluation
from mimirax.inference.optimizers import OptaxOptimizer, adam
from mimirax.inference.resampling import DataResampling
from mimirax.inference.vi import MeanFieldVI

__all__ = [
    "AugmentedLagrangian",
    "AugmentedLagrangianResult",
    "HMC",
    "METHODS",
    "NUTS",
    "DataResampling",
    "InferenceProblem",
    "MadeToMeasure",
    "MeanFieldVI",
    "NelderMead",
    "OptaxOptimizer",
    "ProblemEvaluation",
    "adam",
    "augmented_lagrangian_value",
    "defect_violation",
    "made_to_measure",
]

METHODS: Registry[type] = Registry("inference method")
METHODS.register("optax", OptaxOptimizer)
METHODS.register("hmc", HMC)
METHODS.register("nuts", NUTS)
METHODS.register("mean_field_vi", MeanFieldVI)
METHODS.register("nelder_mead", NelderMead)
METHODS.register("made_to_measure", MadeToMeasure)
METHODS.register("data_resampling", DataResampling)
