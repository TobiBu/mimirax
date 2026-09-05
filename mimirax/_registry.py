"""The registry idiom every pluggable family in mimirax uses.

Observables, likelihoods, priors, reparameterizations and inference methods are
all *registries*: a module-level mapping from a name to an implementation, a
``register`` call that refuses a silent overwrite, and an ``available`` query.
This is yggdrax's ``register_tree_builder`` / ``available_tree_types`` pattern
(``yggdrax/tree.py``) made generic, so a reader of one package reads the other.

WHY A REGISTRY AND NOT A BRANCH. Extensibility is the package's primary design
axis: adding an observable or a sampler must not require editing a core module.
A registry gives a new implementation one call to make instead of one ``elif``
to add, and it gives the error message for a typo the full list of what exists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> implementation mapping with explicit registration.

    Parameters
    ----------
    kind : str
        What the registry holds, used in error messages -- ``"observable"``,
        ``"likelihood"``, ...
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @property
    def kind(self) -> str:
        """Return the kind of implementation this registry holds.

        Returns
        -------
        str
            The ``kind`` given at construction.
        """
        return self._kind

    def register(self, name: str, item: T, *, overwrite: bool = False) -> T:
        """Register ``item`` under ``name``.

        Parameters
        ----------
        name : str
            Lookup key. Leading and trailing whitespace is stripped.
        item : T
            The implementation to register.
        overwrite : bool
            Whether an existing entry under ``name`` may be replaced. Off by
            default so two plugins cannot silently shadow each other.

        Returns
        -------
        T
            ``item``, unchanged, so the call can wrap a definition.

        Raises
        ------
        ValueError
            If ``name`` is empty, or already registered and ``overwrite`` is
            ``False``.
        """
        normalized = name.strip()
        if not normalized:
            raise ValueError(f"{self._kind} name must be a non-empty string")
        if normalized in self._items and not overwrite:
            raise ValueError(
                f"{self._kind} '{normalized}' is already registered; "
                "pass overwrite=True to replace it"
            )
        self._items[normalized] = item
        return item

    def decorate(self, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``.

        Parameters
        ----------
        name : str
            Lookup key.
        overwrite : bool
            Passed through to :meth:`register`.

        Returns
        -------
        Callable[[T], T]
            A decorator returning its argument unchanged.
        """

        def _decorator(item: T) -> T:
            return self.register(name, item, overwrite=overwrite)

        return _decorator

    def get(self, name: str) -> T:
        """Look up the implementation registered under ``name``.

        Parameters
        ----------
        name : str
            Lookup key.

        Returns
        -------
        T
            The registered implementation.

        Raises
        ------
        KeyError
            If nothing is registered under ``name``. The message lists every
            registered name.
        """
        item = self._items.get(name.strip())
        if item is None:
            supported = ", ".join(f"'{key}'" for key in self.available())
            raise KeyError(f"Unknown {self._kind} '{name}'. Registered: ({supported})")
        return item

    def available(self) -> tuple[str, ...]:
        """Return the registered names, sorted.

        Returns
        -------
        tuple[str, ...]
            Registered names in sorted order.
        """
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip() in self._items

    def __len__(self) -> int:
        return len(self._items)
