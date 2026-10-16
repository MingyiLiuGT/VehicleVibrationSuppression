import bootstrap
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from core import *

class ScientificTests(unittest.TestCase):
    def test_zoh_against_independent_ode(self):
        rng=np.random.default_rng(42)
        for mass in [240.,300.,420.]:
            a,b,e=continuous(mass);ad,bd,ed=discrete(mass)
            x=rng.normal(0,[.01,.1,.01,.1]);u=400.;r=.02
            sol=solve_ivp(lambda t,s:a@s+b*u+e*r,[0,DT],x,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(ad@x+bd*u+ed*r,sol.y[:,-1],atol=1e-10)
    def test_lifted_cost_and_independent_qp(self):
        rng=np.random.default_rng(123)
        for _ in range(20):
            qp=MPC(rng.uniform(240,420),horizon=10)
            x=rng.normal(0,[.02,.2,.02,.2]);road=rng.normal(0,.02,10);v=rng.uniform(-1,1,10)
            state=x.copy(); yy=[]
            for i in range(10):
                y=outputs(state,500*v[i],road[i],qp.mass)/np.array([1,TRAVEL_SCALE,LOAD_SCALE])
                yy.extend(y*np.sqrt(WEIGHTS));state=qp.ad@state+qp.bd*500*v[i]+qp.ed*road[i]
            np.testing.assert_allclose(yy,qp.gx@x+qp.gu@v+qp.gr@road,atol=1e-10)
            _,opt=qp.solve(x,road);f=qp.fx@x+qp.fr@road
            fun=lambda z:.5*z@qp.h@z+f@z
            ref=minimize(fun,np.zeros(10),jac=lambda z:qp.h@z+f,method='SLSQP',bounds=[(-1,1)]*10,
                         options={'ftol':1e-11,'maxiter':1000})
            self.assertTrue(ref.success);self.assertLess(abs(fun(opt)-fun(ref.x)),1e-4)
            self.assertLess(qp.last['kkt'],2e-5)
            self.assertGreater(np.linalg.eigvalsh(qp.h).min(),0)
    def test_mass_sweep_qps(self):
        rng=np.random.default_rng(20261010)
        errors=[]
        for mass in [240,270,285,300,315,330,360]:
            qp=MPC(mass)
            for _ in range(3):
                x=rng.normal(0,[.005,.04,.003,.03]);road=rng.normal(0,.008,30)
                _,v=qp.solve(x,road);f=qp.fx@x+qp.fr@road
                objective=lambda z:.5*z@qp.h@z+f@z
                ref=minimize(objective,np.zeros(30),jac=lambda z:qp.h@z+f,
                    bounds=[(-1,1)]*30,method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
                self.assertTrue(ref.success)
                errors.append(abs(objective(v)-objective(ref.x)))
        self.assertLess(max(errors),1e-6)
        print(f'21 independent mass QPs: maximum objective difference {max(errors):.3e}')
    def test_rls_convergence_and_no_excitation(self):
        est=MassRLS();x=np.array([.02,.1,0,0]);u=100.
        a=outputs(x,u,0,mass=420)[0]
        for _ in range(1000):est.update(x,u,a)
        self.assertLess(abs(est.mass-420),.01)
        frozen=MassRLS()
        for _ in range(1000): frozen.update(np.zeros(4),0,.1)
        self.assertEqual(frozen.mass,300.)
    def test_force_projection(self):
        from pathlib import Path
        path=Path(__file__).parent/'results/models/policy.npz'
        if not path.exists(): self.skipTest('model not trained yet')
        p=Policy(path)
        for factor in [-100,0,100]:
            u,_=p.action(np.ones(4)*factor,np.ones(30)*factor,300.)
            self.assertLessEqual(abs(u),500.)

if __name__=='__main__':unittest.main()
