from collections import namedtuple
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from abc import abstractmethod
from dataclasses import dataclass

AnalyticalDomain = namedtuple("AnalyticalDomain", ["fx", "fy", "Jf", "range_u", "range_v"])


class Surface:
    """
    A simple class for a surface.
    """

    @abstractmethod
    def __call__(self, q: Tuple[np.float64, np.float64]) -> np.float64:
        """Evaluate the surface at q"""

    @abstractmethod
    def dq(self, q: Tuple[np.float64, np.float64]) -> Tuple[np.float64, np.float64]:
        """Evaluate the derivate with respect to q"""


class Graph(Surface):
    """
    A simple class for a surface.
    Restricted to the graph of a function
    """

    def __init__(
        self,
        f: Callable[[Tuple[np.float64, np.float64]], np.float64],  # The arguments are assumed to be Cartesian coordinates
        f_q: Callable[[Tuple[np.float64, np.float64]], Tuple[np.float64, np.float64]],  # The arguments are assumed to be Cartesian coordinates
    ):
        self.f = f
        self.f_q = f_q

    def __call__(self, q: Tuple[np.float64, np.float64]) -> np.float64:
        return self.f(q)

    def dq(self, q: Tuple[np.float64, np.float64]) -> Tuple[np.float64, np.float64]:
        """Evaluate the derivate with respect to q"""
        return self.f_q(q)


@dataclass
class ParameterManager:
    uv: Tuple[np.float64, np.float64]
    q_old: Tuple[np.float64, np.float64] = None

    def _compute_parameters_from_xyposition(self, q: Tuple[np.float64, np.float64], parametric_domain: AnalyticalDomain):
        if all(q == self.q_old):
            self.q_old = q
            return
        uv0 = self.uv
        g: NDArray[np.float64] = np.array([parametric_domain.fx(*uv0), parametric_domain.fy(*uv0)]) - q
        i = 0
        while np.linalg.norm(g) > 1e-12:
            i += 1
            if i > 10:
                print(i, g, uv0)
            Jg: NDArray[np.float64] = parametric_domain.Jf(*uv0)
            g: NDArray[np.float64] = np.array([parametric_domain.fx(*uv0), parametric_domain.fy(*uv0)]) - q
            delta = np.array((np.matrix(Jg) ** -1).T @ g)[0]
            uv0 -= delta
        self.uv = uv0
        self.q_old = q


class ParametricSurface(Surface):
    """
    A simple class for a surface.
    Restricted to the graph of a function
    """

    def __init__(
        self,
        parametric_domain: AnalyticalDomain,
        f: Callable[[Tuple[np.float64, np.float64]], np.float64],  # The arguments are assumed to be parameters according to parametric_domain
        f_q: Callable[
            [Tuple[np.float64, np.float64]], Tuple[np.float64, np.float64]
        ],  # The arguments are assumed to be parameters according to parametric_domain
    ):
        self.parametric_domain = parametric_domain
        self.parameter_manager: ParameterManager = None
        self.f = f
        self.f_q = f_q

    def __call__(self, q):
        if not self.parameter_manager:
            raise ValueError("Parameter manager is not set.")
        self.parameter_manager._compute_parameters_from_xyposition(q, self.parametric_domain)
        return self.f(self.parameter_manager.uv)

    def dq(self, q):
        """Evaluate the derivate with respect to q"""
        if not self.parameter_manager:
            raise ValueError("Parameter manager is not set.")
        self.parameter_manager._compute_parameters_from_xyposition(q, self.parametric_domain)
        return np.array(np.matrix(self.parametric_domain.Jf(*self.parameter_manager.uv)) ** -1 @ self.f_q(self.parameter_manager.uv))[0]

    def _compute_parameters_from_xyposition(self, q: Tuple[np.float64, np.float64], uv0: Tuple[np.float64, np.float64]):
        g: NDArray[np.float64] = np.array([self.parametric_domain.fx(*uv0), self.parametric_domain.fy(*uv0)]) - q
        i = 0
        while np.linalg.norm(g) > 1e-12:
            i += 1
            if i > 10:
                print(i, g, uv0)
            Jg: NDArray[np.float64] = self.parametric_domain.Jf(*uv0)
            g: NDArray[np.float64] = np.array([self.parametric_domain.fx(*uv0), self.parametric_domain.fy(*uv0)]) - q
            delta = np.array((np.matrix(Jg) ** -1).T @ g)[0]
            uv0 -= delta
        return uv0


circular_analytical_domain = AnalyticalDomain(
    lambda u, v: u * np.cos(v),
    lambda u, v: u * np.sin(v),
    lambda u, v: np.matrix([[np.cos(v), np.sin(v)], [-u * np.sin(v), u * np.cos(v)]]),
    [0.5, 2.5],
    [0, 2 * np.pi],
)

unit_square = AnalyticalDomain(
    lambda u, v: u,
    lambda u, v: v,
    lambda u, v: np.matrix([[1, 0], [0, 1]]),
    [-1, 1],
    [-1, 1],
)
