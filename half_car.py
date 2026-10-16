"""Four-DOF heave/pitch half-car and two-actuator constrained preview MPC.

Coordinates q=[z, theta, z_uf, z_ur], x=[q, qdot], positive upward;
positive pitch raises the front. Parameters retain the original body geometry.
Wheel and tire parameters match the revised quarter-car at each axle.
"""
import bootstrap
import numpy as np
from scipy.linalg import expm
from scipy import sparse
import osqp
from core import DT, HORIZON, KS, CS, KT, MU, FORCE_SCALE, EFFORT

MS, LENGTH = 915., 2.825
ARM = LENGTH / 2
INERTIA = MS * LENGTH**2 / 12
MASS = np.diag([MS, INERTIA, MU, MU])
S = np.array([[1., ARM, -1., 0.], [1., -ARM, 0., -1.]])
STATIC_LOAD = (MS / 2 + MU) * 9.81
WEIGHTS = np.array([1., 1., .02, .02, .02, .02])

def continuous():
    k = KS * S.T @ S + np.diag([0., 0., KT, KT])
    c = CS * S.T @ S
    a = np.zeros((8, 8)); a[:4, 4:] = np.eye(4)
    a[4:, :4] = -np.linalg.solve(MASS, k)
    a[4:, 4:] = -np.linalg.solve(MASS, c)
    b = np.zeros((8, 2)); b[4:] = np.linalg.solve(MASS, S.T)
    e = np.zeros((8, 2)); e[6:] = np.eye(2) * KT / MU
    return a, b, e

def discrete():
    a, b, e = continuous()
    aug = np.zeros((12, 12)); aug[:8, :8] = a
    aug[:8, 8:10] = b; aug[:8, 10:] = e
    z = expm(aug * DT)
    return z[:8, :8], z[:8, 8:10], z[:8, 10:]

def outputs(x, u, road):
    """Heave/pitch accelerations, two travels, and two dynamic tire forces."""
    force = -KS * (S @ x[:4]) - CS * (S @ x[4:]) + u
    return np.r_[force.sum() / MS, ARM * (force[0] - force[1]) / INERTIA,
                 S @ x[:4], KT * (x[2:4] - road)]

class HalfMPC:
    def __init__(self, horizon=HORIZON):
        self.n = horizon
        self.ad, self.bd, self.ed = discrete()
        a, b, _ = continuous()
        c = np.zeros((6, 8)); c[0] = a[4]; c[1] = ARM * a[5]
        c[2:4, :4] = S / .08; c[4, 2] = c[5, 3] = KT / STATIC_LOAD
        d = np.zeros((6, 2)); d[0] = b[4] * FORCE_SCALE
        d[1] = ARM * b[5] * FORCE_SCALE
        er = np.zeros((6, 2)); er[4:] = -np.eye(2) * KT / STATIC_LOAD
        px = np.eye(8); pu = np.zeros((8, 2*horizon)); pr = pu.copy()
        gx, gu, gr = [], [], []
        for i in range(horizon):
            sl = slice(2*i, 2*i+2)
            ux = c @ pu; ux[:, sl] += d
            rx = c @ pr; rx[:, sl] += er
            gx.append(c @ px); gu.append(ux); gr.append(rx)
            px = self.ad @ px; pu = self.ad @ pu; pr = self.ad @ pr
            pu[:, sl] += self.bd * FORCE_SCALE; pr[:, sl] += self.ed
        w = np.tile(np.sqrt(WEIGHTS), horizon)
        self.gx = np.vstack(gx)*w[:, None]
        self.gu = np.vstack(gu)*w[:, None]
        self.gr = np.vstack(gr)*w[:, None]
        self.h = 2*(self.gu.T @ self.gu + EFFORT*np.eye(2*horizon))
        self.fx = 2*self.gu.T @ self.gx; self.fr = 2*self.gu.T @ self.gr
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(self.h)), q=np.zeros(2*horizon),
            A=sparse.eye(2*horizon, format='csc'), l=-np.ones(2*horizon),
            u=np.ones(2*horizon), verbose=False, eps_abs=1e-7, eps_rel=1e-7,
            max_iter=20000, polishing=False, adaptive_rho=True, check_termination=10)

    def solve(self, x, road):
        f = self.fx @ x + self.fr @ np.asarray(road).reshape(-1)
        self.solver.update(q=f)
        result = self.solver.solve(raise_error=True)
        if result.info.status != 'solved':
            raise RuntimeError(result.info.status)
        v = result.x
        self.kkt = float(np.max(abs(v-np.clip(v-(self.h@v+f), -1, 1))))
        return np.clip(v[:2]*FORCE_SCALE, -500, 500), v.copy()

def validate():
    """Independent force balance, integration, prediction and QP checks."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize
    rng = np.random.default_rng(20261011)
    a, b, e = continuous(); ad, bd, ed = discrete(); qp = HalfMPC()
    errors = []
    for _ in range(8):
        x = rng.normal(0, [.01, .003, .008, .008, .05, .02, .05, .05])
        u = rng.uniform(-500, 500, 2); r = rng.normal(0, .01, 2)
        delta = np.array([x[0]+ARM*x[1]-x[2], x[0]-ARM*x[1]-x[3]])
        velocity = np.array([x[4]+ARM*x[5]-x[6], x[4]-ARM*x[5]-x[7]])
        f = -KS*delta-CS*velocity+u
        expected = np.r_[x[4:], (f[0]+f[1])/MS, ARM*(f[0]-f[1])/INERTIA,
                         (-f[0]-KT*(x[2]-r[0]))/MU, (-f[1]-KT*(x[3]-r[1]))/MU]
        np.testing.assert_allclose(a@x+b@u+e@r, expected, atol=1e-12)
        sol = solve_ivp(lambda t, s: a@s+b@u+e@r, [0, DT], x,
                        rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(ad@x+bd@u+ed@r, sol.y[:, -1], atol=2e-10)
        road = rng.normal(0, .008, (HORIZON, 2)); v = rng.uniform(-1, 1, 2*HORIZON)
        state = x.copy(); yy = []
        scale = [1, 1/ARM, .08, .08, STATIC_LOAD, STATIC_LOAD]
        for k in range(HORIZON):
            force = 500*v[2*k:2*k+2]
            yy.extend(outputs(state, force, road[k])/scale*np.sqrt(WEIGHTS))
            state = ad@state+bd@force+ed@road[k]
        np.testing.assert_allclose(yy, qp.gx@x+qp.gu@v+qp.gr@road.ravel(), atol=1e-10)
        _, opt = qp.solve(x, road); f = qp.fx@x+qp.fr@road.ravel()
        fun = lambda v: .5*v@qp.h@v+f@v
        ref = minimize(fun, np.zeros(2*HORIZON), jac=lambda v: qp.h@v+f,
                       bounds=[(-1, 1)]*(2*HORIZON), method='SLSQP',
                       options={'ftol':1e-11, 'maxiter':2000})
        assert ref.success, ref.message
        errors.append(abs(fun(opt)-fun(ref.x)))
    assert max(errors) < 2e-5
    assert np.linalg.eigvalsh(qp.h).min() > 0
    return dict(independent_qps=8, maximum_objective_difference=max(errors),
                force_balance=True, zoh_vs_ode=True, lifted_predictions=True)

if __name__ == '__main__':
    print(validate())
