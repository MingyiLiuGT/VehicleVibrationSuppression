"""Development and held-out mass adaptation optimization. Original results untouched."""
import bootstrap
from bootstrap import ROOT
import core
import numpy as np,json,csv,argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
OUT=ROOT/'results/mass_optimization';OUT.mkdir(exist_ok=True)
MASSES=[200.,240.,360.,420.]
WINDOWS={'loaded':(4,20),'early':(4,8),'settled':(8,20),'whole':(0,24),'recovery':(20,24)}
def savecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protocol():
 p=dict(training_mass_range_kg=[200,450],development_seeds={'sine':list(range(15000,15004)),'multisine':list(range(15100,15104))},heldout_seeds={'sine':list(range(16000,16012)),'multisine':list(range(16100,16112))},loads_kg=MASSES,duration_s=24,change_times_s=[4,20],windows_s=WINDOWS,primary='Loaded-segment acceleration RMS; all load/road cases retained; also travel, tire, force and contact.',selection='Training/action-validation data and development roads only. Held-out roads frozen until model selection.',mass_update='Causal prior-sample estimate; no true change time available to controller.',baseline='Same policy with mass input fixed at 300 kg; also previous policy and matched MPC.',objective='Improve actual control and mass-dependent action accuracy, never maximize relative gain by degrading the fixed-input baseline.')
 target=OUT/'protocol.json'
 if target.exists():assert json.loads(target.read_text())==json.loads(json.dumps(p))
 else:target.write_text(json.dumps(p,indent=2))
 return p
