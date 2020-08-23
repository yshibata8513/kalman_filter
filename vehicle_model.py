import torch
import numpy as np
import sys
sys.path.append("./src")
from models import *



#state = [x,y,yaw,pitch,roll,vx,vy,r]
#control = [ay,yr]
class Vehicle_Model(BasePlantModel):
    def __init__(self):
        super().__init__()
        self.dim_state = 8
        self.dim_control = 1


    def CalcFrontTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]
        Cf = self.cf_calculator(state,control)
        return Cf*(delta - (vy+self.Lf*r)/vx)

    def CalcRearTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]
        Cr = self.cr_calculator(state,control)
        return Cr*(-vy + self.Lr*r)/vx

    def CalcLateralForce(self,state,control):
        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        return Ff_y + Fr_y 

    def CalcDsDt(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        phi = state[...,p_phi]
        bias = state[...,p_bias]
        delta = control[...,p_delta]

        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        
        dx_dt = vx*cos(yaw) - vy*sin(yaw)
        dy_dt = vx*sin(yaw) + vy*cos(yaw)
        dvx_dt = r*vy        
        dvy_dt = -r*vx + (Ff_y + Fr_y)/self.M
        dr_dt = (self.Lf*Ff_y - self.Lr*Fr_y)/self.Iz
        dphi_dt = torch.zeros_like(dx_dt)
        dbias_dt = torch.zeros_like(dx_dt)
        return torch.cat([dx_dt,dy_dt,dvx_dt,dvy_dt,dr_dt,dphi_dt,dbias_dt],dim=-1)


    def forward(self,state,control,dt):
        vy = state[...,p_vy]
        r = state[...,p_r]
        phi = state[...,p_phi]
        bias = state[...,p_bias]

        ds_dt = self.CalcDsDt(state,control).view(-1,self.dim_state)

        return state.view(-1,self.dim_state) + ds_dt*dt.view(-1,1)        


g = 9.8

class KinematicObserver(BaseObserverModel):
    def __init__(self):
        super().__init__()


    def CalcFrontTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]

        return self.Cf*(delta - (vy+self.Lf*r)/vx)

    def CalcRearTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]

        return self.Cr*(-vy + self.Lr*r)/vx

    def forward(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        sin_phi = state[...,p_sin_phi]
        bias = state[...,p_bias]
        delta = control[...,p_delta]
 
        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        
        dvy_dt = -r*vx - g*sin_phi + (Ff_y + Fr_y)/self.M

        gy = (dvy_dt + r*vx + bias).view(-1,1)
        r = r.view(-1,1)
        
        return torch.cat([x,y,yaw,gy,r],dim=-1)
        

#state = [x,y,yaw,vx,vy,r,sin,d]
#control = [ay,yr]
class Vehicle_Model(BasePlantModel):
    def __init__(self):
        super().__init__()
        self.dim_state = 8
        self.dim_control = 1


    def CalcFrontTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]

        return self.Cf*(delta - (vy+self.Lf*r)/vx)

    def CalcRearTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]
        #Cr = self.CalcCr()

        return self.Cr*(-vy + self.Lr*r)/vx

    def CalcDsDt(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        phi = state[...,p_phi]
        bias = state[...,p_bias]
        delta = control[...,p_delta]

        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        
        dx_dt = vx*cos(yaw) - vy*sin(yaw)
        dy_dt = vx*sin(yaw) + vy*cos(yaw)
        dvx_dt = r*vy        
        dvy_dt = -r*vx + (Ff_y + Fr_y)/self.M
        dr_dt = (self.Lf*Ff_y - self.Lr*Fr_y)/self.Iz
        dphi_dt = torch.zeros_like(dx_dt)
        dbias_dt = torch.zeros_like(dx_dt)
        return torch.cat([dx_dt,dy_dt,dvx_dt,dvy_dt,dr_dt,dphi_dt,dbias_dt],dim=-1)


    def forward(self,state,control,dt):
        vy = state[...,p_vy]
        r = state[...,p_r]
        phi = state[...,p_phi]
        bias = state[...,p_bias]

        ds_dt = self.CalcDsDt(state,control).view(-1,self.dim_state)

        return state.view(-1,self.dim_state) + ds_dt*dt.view(-1,1)        


g = 9.8

class Vehicle_Observer(BaseObserverModel):
    def __init__(self):
        super().__init__()


    def CalcFrontTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]

        return self.Cf*(delta - (vy+self.Lf*r)/vx)

    def CalcRearTireForce(self,state,control):
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        delta = control[...,p_delta]

        return self.Cr*(-vy + self.Lr*r)/vx

    def forward(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        sin_phi = state[...,p_sin_phi]
        bias = state[...,p_bias]
        delta = control[...,p_delta]
 
        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        
        dvy_dt = -r*vx - g*sin_phi + (Ff_y + Fr_y)/self.M

        gy = (dvy_dt + r*vx + bias).view(-1,1)
        r = r.view(-1,1)
        
        return torch.cat([x,y,yaw,gy,r],dim=-1)
        







