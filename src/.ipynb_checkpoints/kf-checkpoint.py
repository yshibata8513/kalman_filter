
import torch
import torch.nn as nn
import numpy as np


class KalmanFilterCalculator(nn.Module):
    def __init__(self,plant,observer):
        super().__init__()
        self.plant = plant
        self.observer = observer
    
    def _prediction(self,_state,control,dt,_P):
        state_ = self.plant(_state,control,dt)
        F = self.plant.F(_state,control,dt)
        Q = self.plant.Q
        P_ = F@_P@F.T + Q
        return state_,P_
    
    def _filtering(self,_state,dt,_P,obs):
        
        H = self.observer.H(_state)
        R = self.observer.R
        
        K = _P@H.T@(H@_P@H.T + R).inverse()

        obs_pred = self.observer(_state)
        
        state_ = _state + K@(obs - obs_pred)
        P_ = _P - K@H@_P
        
        return state_,P_
        
        
    def _smoothing(self,state_p,state_f,_state_s,control,P_p,P_f,_P_s,dt):
        F = self.plant.F(state_f,control,dt)
        A = P_f@F.T@P_p.inverse()
        state_s_ = state_f +A@(_state_s - state_p)
        P_s_ = P_f + A@(_P_s - P_p)@A.T
        return state_s_,P_s_


class KalamanFilter:
    def __init__(self,calculator):
        self.calculator = calculator
        
    def filtering(self,s_init,P_init,dts,observations,controls=None):
        Ss_p = [s_init.view(-1,1)]
        Ss_f = [s_init.view(-1,1)]
        Ps_p = [P_init]
        Ps_f = [P_init]
        p = P_init
        s = s_init
        T = len(observations)
        if controls == None:
          controls = torch.zeros(T)
        for t in range(T):
            dt = dts[t]
            c = controls[t]
            obs = observations[t].view(-1,1)
            s,p = self.calculator._prediction(s.view(-1,1),c.view(-1,1),dt,p)
            Ss_p.append(s.data.clone())
            Ps_p.append(p.data.clone())
            s,p = self.calculator._filtering(s.view(-1,1),dt,p,obs)
            Ss_f.append(s.data.clone())
            Ps_f.append(p.data.clone())
        Ss_p = torch.cat([s.view(1,-1) for s in Ss_p])
        Ss_f = torch.cat([s.view(1,-1) for s in Ss_f])
        h = w = Ps_p[0].size()[0]
        Ps_p = torch.cat([s.view(1,h,w) for s in Ps_p])
        Ps_f = torch.cat([s.view(1,h,w) for s in Ps_f])
        return Ss_p,Ps_p,Ss_f,Ps_f

    def smoothing(self,Ss_p,Ss_f,Ps_p,Ps_f,dts,controls=None):
        s = Ss_f[-1]
        p = Ps_f[-1]
        Ss_s = [Ss_f[-1]]
        Ps_s = [Ps_f[-1]]
        T = len(Ss_p) - 1
        if controls == None:
          controls = torch.zeros(T)
        for _t in range(T):
            t = T - _t 
            dt = dts[t-1]
            s_p = Ss_p[t]
            s_f = Ss_f[t-1]
            P_p = Ps_p[t]
            P_f = Ps_f[t-1]
            c = controls[t-1]
            s,p = self.calculator._smoothing(s_p.view(-1,1),s_f.view(-1,1),s.view(-1,1),c.view(-1,1),P_p,P_f,p,dt)
            Ss_s.append(s.data.clone())
            Ps_s.append(p.data.clone())
        Ss_s = torch.Tensor(np.array(torch.cat([s.view(1,-1) for s in Ss_s]))[::-1].copy())
        h = w = Ps_p[0].size()[0]
        Ps_s = torch.Tensor(np.array(torch.cat([s.view(1,h,w) for s in Ps_s]))[::-1].copy())
        return Ss_s,Ps_s