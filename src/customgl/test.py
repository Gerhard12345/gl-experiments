import numpy as np
from rattle.rattle import AnalyticalDomain, ParametricSurface, ParameterManager

a = AnalyticalDomain(
    lambda r, phi: r * np.cos(phi),
    lambda r, phi: r * np.sin(phi),
    lambda r, phi: np.array([[np.cos(phi), np.sin(phi)], [-r * np.sin(phi), r * np.cos(phi)]]),
    [0, 1],
    [0, 2 * np.pi],
)
pm = ParameterManager(uv=(0, 0))

p = ParametricSurface(a, pm, lambda r, phi: -0.25 * phi + (r - 1) ** 2, lambda r, phi: [2 * (r - 1), -0.25])


tot_vals = []
for phi in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
    vals = []
    for r in np.arange(0.5, 1.6, 0.1):
        rphi0 = (r, phi)
        xy = [a.fx(*rphi0), a.fy(*rphi0)]
        rphi = (rphi0[0] - 0.2, rphi0[1] + 0.2)
        p.parameter_manager.uv = rphi
        z = p(xy)
        rphi = p.parameter_manager.uv
        z2 = p.f(*rphi)
        print(z, z2, z - z2)
        vals.append([a.fx(*rphi), a.fy(*rphi), z2])
    tot_vals.append(vals)


zs1 = []
zs2 = []
for vals in tot_vals[0]:
    zs1.append(vals[2])
for vals in tot_vals[3]:
    zs2.append(vals[2])

rphi2 = [1.5, 0]

for phi, x, y in zip([0, np.pi / 2, np.pi, 3 * np.pi / 2], [1.5, 0, -1.5, 0], [0, 1.5, 0, -1.5]):
    rphi2[1] = phi
    xy = [a.fx(*rphi2), a.fy(*rphi2)]
    rphi = (rphi2[0] - 0.2, rphi2[1] + 0.2)
    p.parameter_manager.uv = rphi
    dq = p.dq(xy)
    rphi2 = p.parameter_manager.uv
    print(dq)
    print(np.matrix(p.parametric_domain.Jf(*rphi2)) ** -1 @ p.f_q(*rphi2))
    print(
        [
            2 * x * (1 - 1 / np.sqrt(x**2 + y**2)) + 0.25 * y / (x**2 + y**2),
            2 * y * (1 - 1 / np.sqrt(x**2 + y**2)) - 0.25 * x / (x**2 + y**2),
        ]
    )
    print()

print()
p = ParametricSurface(a, pm, lambda r, phi: -(phi**2) + (r - 1) ** 2, lambda r, phi: [2 * (r - 1), -2 * phi])
rphi2 = [1.5, 0]

for phi, x, y in zip([0, np.pi / 2, np.pi, 3 * np.pi / 2], [1.5, 0, -1.5, 0], [0, 1.5, 0, -1.5]):
    rphi2[1] = phi
    xy = [a.fx(*rphi2), a.fy(*rphi2)]
    rphi = (rphi2[0] - 0.2, rphi2[1] + 0.2)
    p.parameter_manager.uv = rphi
    dq = p.dq(xy)
    rphi2 = p.parameter_manager.uv
    print(dq)
    print(np.matrix(p.parametric_domain.Jf(*rphi2)) ** -1 @ p.f_q(*rphi2))
    print(
        [
            2 * x * (1 - 1 / np.sqrt(x**2 + y**2)) + 2 * np.mod(np.arctan2(y, x), 2 * np.pi) * y / (x**2 + y**2),
            2 * y * (1 - 1 / np.sqrt(x**2 + y**2)) - 2 * np.mod(np.arctan2(y, x), 2 * np.pi) * x / (x**2 + y**2),
        ]
    )
    print()
