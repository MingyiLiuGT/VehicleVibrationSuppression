"""Independent estimator diagnostic under prescribed excitation.

This is not an extra controller or a claimed comfort improvement. It separates
RLS tracking capability from loss of excitation in the controlled plant.
"""
import bootstrap
from bootstrap import ROOT
import csv
import numpy as np
from core import *

def main():
    out=ROOT/'results';rows=[]
    for noise in [0.,.05,.1]:
        for rep in range(8):
            seed=7000+rep; rng=np.random.default_rng(seed)
            tt,road=road_profile('multisine',seed,duration=12.)
            est=MassRLS();x=np.zeros(4);all_rows=[]
            plants={m:discrete(m) for m in [300.,420.]}
            for k in range(1200):
                t=tt[k];mass=420. if 4<=t<8 else 300.
                u=100*np.sin(2*np.pi*1.3*t);y=outputs(x,u,road[k],mass)
                all_rows.append([t,mass,est.mass,y[0],u,est.updates])
                est.update(x,u,y[0]+rng.normal(0,noise))
                a,b,e=plants[mass];x=a@x+b*u+e*road[k]
            arr=np.array(all_rows);err=100*abs(arr[:,2]-arr[:,1])/arr[:,1]
            tail=((arr[:,0]>=7)&(arr[:,0]<8))|(arr[:,0]>=11)
            delays=[]
            for start,end in [(400,800),(800,1200)]:
                hits=[(k-start)*DT for k in range(start,end-49) if np.all(err[k:k+50]<5)]
                delays.append(hits[0] if hits else '')
            rows.append(dict(noise_sd=noise,seed=seed,tail_mass_error_pct=float(err[tail].mean()),
                       load_convergence_s=delays[0],unload_convergence_s=delays[1],updates=est.updates))
            np.savez_compressed(out/'traces'/f'id_diagnostic_{noise}_{seed}.npz',data=arr)
    with (out/'identification_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
