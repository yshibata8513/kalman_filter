
import torch
import torch.nn as nn
import numpy as numpy

class BasePlantModel(nn.Module):
    
    def __init__(self,use_control,Q):
        super().__init__()
        self.use_control = use_control
        self._Q = Q
    
    def forward(self,**argv):
        raise NotImplementedError()
        
    def F(self,state,control,dt):
        _state = torch.nn.Parameter(state.data.clone())
        if self.use_control:
            control = control.data.clone()
        state_ = self.forward(_state,control,dt)
        F = torch.cat([torch.autograd.grad(outputs=s_, inputs=_state, create_graph=False, retain_graph=True)[0].view(1,-1) for s_ in state_],dim=0)
        return F
    
    def B(self,state,control,dt):
        if self.use_control==False:
            return None
        _state = state.data.clone()
        control = torch.nn.Parameter(control.data.clone())
        state_ = self.forward(_state,control,dt)
        B = torch.cat([torch.autograd.grad(outputs=c, inputs=_state, create_graph=False, retain_graph=True)[0].view(1,-1) for c in control],dim=0)
        return B

    @property
    def Q(self):
        return self._Q.data.clone()


class BaseObserverModel(nn.Module):
    
    def __init__(self,R):
        super().__init__()
        self._R = R
    
    def forward(self,**argv):
        raise NotImplementedError()
        
    def H(self,state):
        _state = torch.nn.Parameter(state.data.clone())
        obs = self.forward(_state)
        H = torch.cat([torch.autograd.grad(outputs=o, inputs=_state, create_graph=False, retain_graph=True)[0].view(1,-1) for o in obs],dim=0)
        return H
    
    @property
    def R(self):
        return self._R.data.clone()

