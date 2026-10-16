"""Optimized mass-conditioned NN and causal RLS for the original quarter-car model.

No changes to stiffness, damping, actuator dynamics, or gravitational loading.
Online operation interpolates frozen neural action heads using estimated mass.
"""
import bootstrap
from bootstrap import ROOT
import numpy as np
from core import KS,CS
DEFAULT_POLICY=ROOT/'results/mass_optimization/mass_heads_final/policy.npz'
class MassConditionedPolicy:
 def __init__(self,path=DEFAULT_POLICY):
  z=np.load(path);self.mean=z['mean'];self.scale=z['scale'];self.grid=z['grid'];self.w=[z[f'w{i}'] for i in range(3)];self.b=[z[f'b{i}'] for i in range(3)]
 def heads(self,xx):
  h=(xx-self.mean)/self.scale
  for w,b in zip(self.w[:-1],self.b[:-1]):h=np.maximum(h@w+b,0)
  return (h@self.w[-1]+self.b[-1])*500
 def predict(self,features):
  xx=np.atleast_2d(features);hh=self.heads(xx[:,:-1]);m=np.clip(xx[:,-1],self.grid[0],self.grid[-1]);hi=np.clip(np.searchsorted(self.grid,m,side='right'),1,len(self.grid)-1);lo=hi-1;alpha=(m-self.grid[lo])/(self.grid[hi]-self.grid[lo]);ii=np.arange(len(xx))
  return (1-alpha)*hh[ii,lo]+alpha*hh[ii,hi]
 def action(self,x,road,mass):
  hh=self.heads(np.r_[x,road]);raw=float(np.interp(mass,self.grid,hh));return float(np.clip(raw,-500,500)),raw
class MassEstimator:
 def __init__(self,forget=.97,gate=.02):self.theta=1.;self.p=100.;self.forget=forget;self.gate=gate;self.updates=0
 @property
 def mass(self):return 300/self.theta
 def update(self,x,u,measured_acc):
  phi=(-KS*(x[0]-x[2])-CS*(x[1]-x[3])+u)/300
  if abs(phi)<self.gate:return self.mass
  gain=self.p*phi/(self.forget+phi*phi*self.p)
  self.theta=float(np.clip(self.theta+gain*(measured_acc-phi*self.theta),300/480,300/180));self.p=float(np.clip((self.p-gain*phi*self.p)/self.forget,1e-8,1e4));self.updates+=1
  return self.mass
