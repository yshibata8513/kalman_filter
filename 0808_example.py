import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import  sys
#sys.path.append("./src")
from models import *
from kf import *

class Kinematic_1d(BasePlantModel):
    def __init__(self):
        super().__init__(use_control=True,Q=torch.FloatTensor([[0.01,0],[0.,0.01]]))
        self.dim_state = 2
        self.dim_control = 1
        self._F = torch.FloatTensor([[0,1],[0,0]])
        self._B = torch.FloatTensor([0,1]).view(2,1)
        #self._Q = torch.FloatTensor([[0.01,0],[0.,0.01]])
        
    def forward(self,state,control,dt):
        F = self.F_(state,control,dt)
        B = self.B_(state,control,dt)
        return F@state + B@control
        
    #@property
    def F_(self,state,control,dt):
        return torch.eye(2) + self._F*dt
        
    #@property
    def B_(self,state,control,dt):
        return self._B*dt
 
class ObsPosition_1d(BaseObserverModel):
    def __init__(self,R=torch.FloatTensor([0.5])):
        super().__init__(R)
        self.dim_state = 2
        self.dim_obs = 1
        self._H = torch.FloatTensor([1,0]).view(1,-1)
        #self._R = torch.FloatTensor([0.5])
        
    @property
    def R(self):
        return self._R
    
    def forward(self,state):
        return self._H@state


dt = 0.5
x =  torch.arange(100)*dt + torch.randn(100)*0.5
x[0] = 0
v = torch.ones(100).float()
states = torch.stack([x,v],dim=-1)
controls = torch.zeros_like(x)
observations = states[1:,0]

P0 = torch.FloatTensor([[0.01,0],[0.,0.01]])
s0 = states[0]
c0 = controls[0]

p_model = Kinematic_1d()
o_model = ObsPosition_1d()
kfc = KalmanFilterCalculator(p_model,o_model)
kf = KalamanFilter(kfc)
dts = torch.ones(99)*0.5

Ss_p,Ps_p,Ss_f,Ps_f = kf.filtering(s0,P0,dts,observations)
Ss_s,Ps_s =  kf.smoothing(Ss_p=Ss_p,Ps_p=Ps_p,Ss_f=Ss_f,Ps_f=Ps_f,dts=dts)

plt.plot(Ss_f[1:21,0])
plt.plot(Ss_s[1:21,0])
plt.plot(observations[:20])







