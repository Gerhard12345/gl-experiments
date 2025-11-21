r"""
This module is used to investigate the rattle algorithm for a
hamiltonian system with algebraic condition. For a separable
hamiltonian, we use the rattle's definition of p_n+1/2, q_n+1
in order to reduce the first step to a scalar problem which is
finding the lambda. The second step can be explicitly solved in
case of a separable hamiltonian.

rattle step 1:
**************

p_ = p_n - 0.5 * h * [ H_q(p_, q_n) - lambda * g_q(q_n) ]
q_next = q_n + 0.5 * h * [ H_p(p_, q_n) + H_p(p_, q_next) ]
0 = g(q_next)

with H(p, q) = 1/(2m)p + m*g*q ==> H_q(p,q) = m*g and H_p(p,q) = 1/m*p. Thus,

p_ = p_n - 0.5 * h * [ m*g - lambda * g_q(q_n) ]
q_next = q_n + h * 1/m*p
0 = g(q_next)

Thus,
p_ = p_n - 0.5 * h * [ m*g - lambda * g_q(q_n) ]
q_next = q_n + h * 1/m * ( p_n - 0.5 * h * [ m*g - lambda * g_q(q_n) ] )
0 = g(q_next)

Solve
0 = g(q_next(lambda)), where
q_next(lambda) = q_n + h * 1/m * ( p_n - 0.5 * h * [ m*g - lambda * g_q(q_n) ] )

For newton iteration, also d(g \circ q_next)/dlambda is required:

dg                   dq_
-- (q_next(lambda)) *  ------- (lambda)  =
dq                 dlambda

g_q (q_next(lambda)) *  h  * 0.5 * h * g_q(q_n) =

0.5 * h^2 * g_q(q_next(lambda)) *  g_q(q_n)

rattle step 2:
**************
p_next = p_ - 0.5 * h * [ m*g + mu * g_q(q_next)]
0 = g_q(q_next) * 1/m*p_next

Using H(p,q) = 1/(2m)p + mg results in

p_next = p_ - 0.5 * h * [ m*g + mu * g_q(q_next)]

0 = g_q(q_next) * 1/m*(p_ - 0.5 * h * [ m*g + mu * g_q(q_next)])

<=>

mu  = [2 / h * g_q(q_next) * (1/m*(p_) - 0.5 * h *  m*g)] / || g_q(q_next) ||**2

"""

import time
import numpy as np
from typing import Tuple
from typing import Callable
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from abc import abstractmethod
from customgl.objects.surface import AnalyticalDomain


class Surface:
    """
    A simple class for a surface.
    Restricted to the graph of a function
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


from dataclasses import dataclass


@dataclass
class ParameterManager:
    uv: Tuple[np.float64, np.float64]


class ParametricSurface(Surface):
    """
    A simple class for a surface.
    Restricted to the graph of a function
    """

    def __init__(
        self,
        parametric_domain: AnalyticalDomain,
        parameter_manager: ParameterManager,
        f: Callable[[Tuple[np.float64, np.float64]], np.float64],  # The arguments are assumed to be parameters according to parametric_domain
        f_q: Callable[
            [Tuple[np.float64, np.float64]], Tuple[np.float64, np.float64]
        ],  # The arguments are assumed to be parameters according to parametric_domain
    ):
        self.parametric_domain = parametric_domain
        self.parameter_manager = parameter_manager
        self.f = f
        self.f_q = f_q

    def __call__(self, q):
        uv = self._compute_parameters_from_xyposition(q, self.parameter_manager.uv)
        self.parameter_manager.uv = uv
        return self.f(self.parameter_manager.uv)

    def dq(self, q):
        """Evaluate the derivate with respect to q"""
        uv = self._compute_parameters_from_xyposition(q, self.parameter_manager.uv)
        self.parameter_manager.uv = uv
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


class AlgebraicCondition:
    """
    Represents an algebraic condition
    for a ball sticked to the graph
    of a function
    """

    def __init__(self, surface: Surface):
        self.surface = surface

    def __call__(self, q: NDArray[np.float64]):
        return q[2] - self.surface(q[0:2])

    def dq(self, q: NDArray[np.float64]):
        """Evaluate the derivate with respect to q"""
        # return -np.array([*self.surface.f_q(q[0:2]), -1])
        return -np.array([*self.surface.dq(q[0:2]), -1])


class Hamiltonian:
    def __init__(self, m: np.float64, algebraic_condition: AlgebraicCondition):
        self.algebraic_condition = algebraic_condition
        self.m = m
        self.g = np.array([0, 0, 9.81])
        self.reflection_damping = 0.5
        self.mtimesg = self.m * self.g

    def __call__(self, p: NDArray[np.float64], q: NDArray[np.float64]):
        return self.m * np.dot(self.g, q) + 0.5 / self.m * np.dot(p, p)


class Rattle:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
        h: np.float64,
    ):
        self.h = h
        self.p = p
        self.q = q
        self.lambda_ = 0.0
        self.E_deformation = 0.0
        self._ps = [self.p]
        self._qs = [self.q]
        self.lambdas = [0.0]
        self.mus = [0.0]
        self.E_deformations = [self.E_deformation]
        self.hamiltonian = hamiltonian
        self.state = "rolling"
        self.algebraic_condition_dq_old = self.hamiltonian.algebraic_condition.dq(self.q)
        self.q_next_base = self.q + self.h * 1 / self.hamiltonian.m * (self.p - 0.5 * self.h * (self.hamiltonian.mtimesg))
        self.h2m = 0.5 * self.h**2 * 1 / self.hamiltonian.m
        self._compute_q_next(self.lambda_)

    @property
    def qs(self):
        return np.array(self._qs)

    @property
    def ps(self):
        return np.array(self._ps)

    def _compute_q_next(
        self,
        lambda_: np.float64,
    ):
        self.q_next = self.q_next_base + lambda_ * self.algebraic_condition_dq_old

    def _g_part1(self):
        return self.hamiltonian.algebraic_condition(self.q_next)

    def _dg_part1(self):
        return np.dot(
            self.algebraic_condition_dq_old,
            self.hamiltonian.algebraic_condition.dq(self.q_next),
        )

    def _newton_iteration(self):
        lambda_new = self.lambda_
        g_lambda = self._g_part1()
        while np.dot(g_lambda, g_lambda) >= 1e-14:
            delta = 1 / self._dg_part1() * g_lambda
            lambda_new -= delta
            self._compute_q_next(lambda_new)
            g_lambda = self._g_part1()
        return lambda_new

    def step(self):
        if self.state == "rolling":
            pnew, qnew, lambda_new, mu_new = self.constraint_step()
            if lambda_new <= -0.1e1:
                self.state = "free"
                pnew, qnew, lambda_new, mu_new = self.free_step()
        elif self.state == "free":
            pnew, qnew, lambda_new, mu_new = self.free_step()
            if self.hamiltonian.algebraic_condition(qnew) < 0:
                pnew, qnew, lambda_new, mu_new = self.bounce_step()
        self.p = pnew
        self.q = qnew
        self.lambda_ = lambda_new
        self._ps.append(pnew)
        self._qs.append(qnew)
        self.lambdas.append(lambda_new)
        self.mus.append(mu_new)
        self.E_deformations.append(self.E_deformation)

    def free_step(self):
        phalf = self.p - 0.5 * self.h * self.hamiltonian.mtimesg
        qnew = self.q + self.h / self.hamiltonian.m * phalf
        pnew = phalf - 0.5 * self.h * self.hamiltonian.mtimesg
        return pnew, qnew, 0.0, 0.0

    def reflect_p_at_collision(self, normal, p, q):
        E0 = self.hamiltonian(p, q)
        normal_component = np.dot(normal, p) * normal / np.dot(normal, normal)
        p = p - (2.0 - self.hamiltonian.reflection_damping) * normal_component
        E1 = self.hamiltonian(p, q)
        self.E_deformation += E0 - E1
        return p

    def bounce_step(self):
        h_old = self.h
        step_size = 0.5 * self.h
        self.h = step_size
        q0 = self.q
        qnew = self.q
        p0 = self.p
        # bisection to find hit point and corresponding h:
        alg_condition = self.hamiltonian.algebraic_condition(qnew)
        while np.abs(alg_condition) >= 1e-14:
            # Restore last valid step:
            step_size *= 0.5
            self.q = q0
            self.p = p0
            pnew, qnew, lambda_new, mu_new = self.free_step()
            alg_condition = self.hamiltonian.algebraic_condition(qnew)
            if alg_condition >= 0:
                self.h += step_size
            else:
                self.h -= step_size
        # hit point found, reflect p at surface normal:
        normal = self.hamiltonian.algebraic_condition.dq(qnew)
        pnew = self.reflect_p_at_collision(normal, pnew, qnew)
        # try to finish step as free step. If not possible,
        # revert and finish as constraint step.
        self.h = h_old - self.h
        self.p = pnew
        self.q = qnew
        pnew, qnew, lambda_new, mu_new = self.free_step()
        if self.hamiltonian.algebraic_condition(qnew) < 0:
            self.p = pnew
            self.q = qnew
            pnew, qnew, lambda_new, mu_new = self.constraint_step()
            self.state = "rolling"
        # finally restore h
        self.h = h_old
        self.p = p0
        self.q = q0
        return pnew, qnew, lambda_new, mu_new

    def constraint_step(self):
        h = self.h
        self.algebraic_condition_dq_old = self.h2m * self.hamiltonian.algebraic_condition.dq(self.q)
        self.q_next_base = self.q + self.h * 1 / self.hamiltonian.m * (self.p - 0.5 * self.h * self.hamiltonian.mtimesg)
        self._compute_q_next(self.lambda_)
        lambda_new = self._newton_iteration()
        phalf = self.hamiltonian.m / self.h * (self.q_next - self.q)
        gradg = self.hamiltonian.algebraic_condition.dq(self.q_next)
        mu_new = 2.0 / h * np.dot(gradg, (phalf - 0.5 * h * self.hamiltonian.mtimesg)) / np.dot(gradg, gradg)
        pnew = phalf - 0.5 * h * (self.hamiltonian.mtimesg + mu_new * gradg)
        return pnew, self.q_next, lambda_new, mu_new


if __name__ == "__main__":
    from customgl.objects.surface import MeshedSurface

    def surface_f(q):
        return (q[0] ** 2 + q[1] ** 2) ** 2

    def surface_df(q):
       return [2 * (q[0] ** 2 + q[1] ** 2) * 2 * q[0], 2 * (q[1] ** 2 + q[0] ** 2) * 2 * q[1]]

    def surface_fx(q):
        return np.sin(0.5 * q[0]) * (3 - 0.3 * q[0])

    def surface_dfx(q):
        return -0.3 * np.sin(0.5 * q[0]) + 0.5 * (3 - 0.3 * q[0]) * np.cos(0.5 * q[0])

    # surface_fy = lambda q: np.exp(-(q[1]/10)**2) + 0.5*np.exp(-((q[1]-25.5)/10)**2)+ 0.5*np.exp(-((q[1]+25.5)/10)**2)
    # surface_dfy = lambda q: -2*q[1]/10*np.exp(-(q[1]/10)**2) - 0.5*2*((q[1]-25.5)/10) * np.exp(-((q[1]-25.5)/10)**2)- 0.5*2*((q[1]+25.5)/10) * np.exp(-((q[1]+25.5)/10)**2)
    def surface_fy(q):
        return 1

    def surface_dfy(q):
        return 0

    def surface_f(q):
        return surface_fx(q) * surface_fy(q) + 0.001 * (q[0] ** 2 + q[1] ** 2)

    def surface_df(q):
        return [
            surface_dfx(q) * surface_fy(q) + 2 * 0.001 * q[0],
            surface_fx(q) * surface_dfy(q) + 0.001 * 2 * q[1],
        ]

    if True:
        # surface_fx=lambda q: 1+0.2*np.sin(0.5*q[0])
        # surface_dfx=lambda q: 0.5*0.2*np.cos(0.5*q[0])
        w = np.array([3, 3])
        mean_v1 = np.array([2, 2])
        mean_v2 = np.array([-2, -2])
        mean_v3 = np.array([-2, 2])
        mean_v4 = np.array([2, -2])

        def atan(q):
            return 3 + 3 * np.arctan(0.1 * np.dot(q, q) - 30)

        def atandq(q):
            return 3 / (1 + (0.1 * np.dot(q, q) - 30) ** 2) * (2 * np.array(q)) * 0.1

        def gaussian(q, mean_v, w):
            return 2 * np.exp(-np.dot((q - mean_v) / w, (q - mean_v) / w))

        def gaussiandq(q, mean_v, w):
            return -2 * (q - mean_v) / w * gaussian(q, mean_v, w)

        def surface_f(q):
            return (
                gaussian(q, mean_v1, 0.5 * w)
                + gaussian(q, mean_v2, 0.5 * w)
                + gaussian(q, mean_v3, 0.5 * w)
                + gaussian(q, mean_v4, 0.5 * w)
                + 1 * atan(0.35 * np.array(q))
            )

        def surface_df(q):
            return (
                lambda q: gaussiandq(q, mean_v1, 0.5 * w)
                + gaussiandq(q, mean_v2, 0.5 * w)
                + gaussiandq(q, mean_v3, 0.5 * w)
                + gaussiandq(q, mean_v4, 0.5 * w)
                + 0.35 * atandq(0.35 * np.array(q))
            )

        # surface_f=lambda q:surface_fx(q)*gaussian(q) + 0.0 * np.dot(q-mean_v,q-mean_v)
        # surface_df=lambda q:[surface_dfx(q)*gaussian(q),0] + surface_fx(q)*gaussiandq(q) + 0.0*(q-mean_v)
    #                     -2*surface_fx(q)*np.exp(-(q[1]-m[0])**2/w**2)* (q[1]-m[0])/w**2,
    #                     -2*surface_fx(q)*np.exp(-(q[1]-m[0])**2/w**2)* (q[2]-m[1])/w**2
    # surface_f = lambda q: (4+np.sin(q[0])) * (3 - 0.3 * q[0])
    # surface_df = lambda q: [(-0.3 * np.sin(q[0]) + (3 - 0.3 * q[0]) * np.cos(q[0])), 0]

    # surface_f = lambda q: q[0] ** 2
    # surface_df = lambda q: np.array([2 * q[0], 0.0])

    # surface_f = lambda q: 0
    # surface_df = lambda q: np.array([0.0, 0.0])
    a = AnalyticalDomain(
        lambda u, v: u * np.cos(v),
        lambda u, v: u * np.sin(v),
        lambda u, v: np.matrix([[np.cos(v), np.sin(v)], [-u * np.sin(v), u * np.cos(v)]]),
        [0, 2],
        [0, 2 * np.pi],
    )
    b = AnalyticalDomain(
        lambda u, v: u,
        lambda u, v: v,
        lambda u, v: np.matrix([[1, 0], [0, 1]]),
        [-24, 24],
        [-24, 24],
    )

    # plt.figure()
    x0 = 8 * np.pi / 2 + 0.2
    y0 = 1.7
    q0 = np.array([x0, y0, surface_f([x0, y0])])
    p0 = np.array([-30, 1, 0])
    my_algebraic_condition = AlgebraicCondition(Surface(surface_f, surface_df))
    my_hamiltonian = Hamiltonian(m=2.0, algebraic_condition=my_algebraic_condition)
    rattle = Rattle(hamiltonian=my_hamiltonian, p=p0, q=q0, h=0.0005)
    t0 = time.time_ns()
    for j in range(50000):
        rattle.step()
    t1 = time.time_ns()
    E = np.array([my_hamiltonian(p, q) for p, q in zip(rattle._ps, rattle._qs)])
    print(f"Computation took {(t1-t0)*1e-9} seconds")
    x = [x[0] for x in rattle.qs]
    y = [x[1] for x in rattle.qs]
    z = [x[2] for x in rattle.qs]
    axes = plt.figure().add_subplot(projection="3d")
    # m = MeshedSurface(analytical_domain=a, f_z = lambda u,v: surface_f([u*np.cos(v),u*np.sin(v)]), h_u=0.05,h_v=0.05)
    m = MeshedSurface(
        analytical_domain=b,
        f_z=lambda u, v: surface_f([u, v]),
        d_f=lambda u, v: surface_df([u, v]),
        h_u=0.0125,
        h_v=0.0125,
        position=np.array([0, 0, 0]),
        material=None,
    )
    # m.plot_surface(axes)
    axes.scatter3D(
        x,
        y,
        z,
        c=np.linalg.norm(rattle.ps, axis=1) - np.min(np.linalg.norm(rattle.ps, axis=1)),
        linewidth=0.5,
        linestyle="-",
        norm=colors.CenteredNorm(),
        cmap="hsv",
    )
    plt.axis("equal")
    plt.figure()
    plt.plot(E + np.array(rattle.E_deformations))
    plt.show()
