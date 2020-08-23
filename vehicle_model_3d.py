import torch
import numpy as np
import sys
sys.path.append("./src")
from models import *
from euler_rotatin import *



p_x = 0
p_y = 1
p_roll = 2
p_pitch = 3
p_yaw = 4

p_vx = 5
p_vy = 6
p_delta = 0
#state = [x,y,yaw,pitch,roll,vx]
#control = [delta]
class KinematicPlantModel(BasePlantModel):
    def __init__(self):
        super().__init__()
        self.dim_state = 6
        self.dim_control = 1

    def CalcYawrate(self,staet,control):
        vx = state[...,p_vx]
        delta = control[...,p_delta]
        return vx/sel.fL*delta 

    def CalcBeta(self,state,control):
        delta = control[...,p_delta]
        return self.Lr/self.L*delta

    def forward(self,state,control,dt):
        ds_dt = self.CalcDsDt(state,control).view(-1,self.dim_state)

        return state.view(-1,self.dim_state) + ds_dt*dt.view(-1,1)       

    def CalcDsDt(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        pitch = state[...,p_pitch]
        roll = state[...,p_roll]
        vx = state[...,p_vx]
        
        delta = control[...,p_delta]

        r = self.CalcYawrate(state,control)
        beta = self.CalcBeta(state,control) 

        vy = vx*torch.tan(beta)

        i2s = EulerRotationXYZ(yaw=yaw,pitch=pitch,roll=roll,train=False)
        
        _gravity = torch.Tensor([0,0,-g*self.M])
        gravity = i2s(_gravity)
        gx,gy = gravity[0],gravity[1]
        
        #自車座標系から慣性座標に変換(xy変位)
        dr_dt_s = torch.stack([vx,vy,torch.zeros_like(vx)],dim=-1)
        dr_dt_i = i2s.inverse(dr_dt_s)
        dx_dt,dy_dt = dr_dt_i[...,0],dr_dt_i[...,1] 

        #自車座標系から慣性座標に変換(回転角速度)
        dyaw_dt_s = r
        dpitch_dt_s = torch.zeros_like(r)
        droll_dt_s =  torch.zeros_like(r)
        dw_dt_s = torch.stack([droll_dt_s,dpitch_dt_s,dyaw_dt_s],dim=-1)
        dw_dt_i = i2s.kinematic_transform(dw_dt_s)
        droll_dt,dpitch_dt,dyaw_dt = dw_dt_i[...,0],dw_dt_i[...,1],dw_dt_i[...,2]

        dvx_dt = r*vy + gx
        dvy_dt = -r*vx + (Ff_y + Fr_y)/self.M + gy
        dr_dt = (self.Lf*Ff_y - self.Lr*Fr_y)/self.Iz
        
        return torch.cat([dx_dt,dy_dt,dyaw_dt,dpitch_dt,droll_dt,dvx_dt],dim=-1)







p_x = 0
p_y = 1
p_roll = 2
p_pitch = 3
p_yaw = 4

p_vx = 5
p_vy = 6
p_delta = 0
#state = [x,y,yaw,pitch,roll,vx,vy,r]
#control = [delta]
class VehiclePlantModel(BasePlantModel):
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

    def CalcDsDt(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        pitch = state[...,p_pitch]
        roll = state[...,p_roll]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        bias = state[...,p_bias]
        delta = control[...,p_delta]

        i2s = EulerRotationXYZ(yaw=yaw,pitch=pitch,roll=roll,train=False)

        Ff_y = self.CalcFrontTireForce(state,control)
        Fr_y = self.CalcRearTireForce(state,control)
        
        _gravity = torch.Tensor([0,0,-g*self.M])
        gravity = i2s(_gravity)
        gx,gy = gravity[0],gravity[1]

        
        #自車座標系から慣性座標に変換(xy変位)
        dr_dt_s = torch.stack([vx,vy,torch.zeros_like(vx)],dim=-1)
        dr_dt_i = i2s.inverse(dr_dt_s)
        dx_dt,dy_dt = dr_dt_i[...,0],dr_dt_i[...,1] 

        #自車座標系から慣性座標に変換(回転角速度)
        dyaw_dt_s = r
        dpitch_dt_s = torch.zeros_like(r)
        droll_dt_s =  torch.zeros_like(r)
        dw_dt_s = torch.stack([droll_dt_s,dpitch_dt_s,dyaw_dt_s],dim=-1)
        dw_dt_i = i2s.kinematic_transform(dw_dt_s)
        droll_dt,dpitch_dt,dyaw_dt = dw_dt_i[...,0],dw_dt_i[...,1],dw_dt_i[...,2]

        dvx_dt = r*vy + gx
        dvy_dt = -r*vx + (Ff_y + Fr_y)/self.M + gy
        dr_dt = (self.Lf*Ff_y - self.Lr*Fr_y)/self.Iz
        
        return torch.cat([dx_dt,dy_dt,dyaw_dt,dpitch_dt,droll_dt,dvx_dt,dvy_dt,dr_dt],dim=-1)


    def forward(self,state,control,dt):
        ds_dt = self.CalcDsDt(state,control).view(-1,self.dim_state)

        return state.view(-1,self.dim_state) + ds_dt*dt.view(-1,1)        


g = 9.8

class KinematicObserver(BaseObserverModel):
    def __init__(self,vehicle_model):
        super().__init__()
        self.vehicle_model = vehicle_model


    def forward(self,state,control):
        x = state[...,p_x]
        y = state[...,p_y]
        yaw = state[...,p_yaw]
        pitch = state[...,p_pitch]
        roll = state[...,p_roll]
        vx = state[...,p_vx]
        vy = state[...,p_vy]
        r = state[...,p_r]
        bias = state[...,p_bias]
        delta = control[...,p_delta]
        M = self.vehicle_model.M

        i2s = EulerRotationXYZ(yaw=yaw,pitch=pitch,roll=roll,train=False)
        
        #自車座標系から慣性座標に変換(方位角)
        _heading = torch.Tensor([1,0,0])
        heading_2d = i2s.inverse(_heading)
        heading_x,heading_y = heading_2d[...,0],heading_2d[...,1]
        heading = torch.atan(heading_y,heading_x)

        Ff_y = self.vehicle_model.CalcFrontTireForce(state,control)
        Fr_y = self.vehicle_model.CalcRearTireForce(state,control)
        Fy = Ff_y + Fr_y  

        _gravity = torch.Tensor([0,0,-g])
        gravity = i2s(_gravity)
        gx,gy = gravity[0],gravity[1]
        ax =  gx.view(1)
        ay =  (Fy/self.M - gy).view(1)
        
        r = r.view(-1,1)
        
        return torch.cat([x,y,heading,ax,ay,r],dim=-1)
        

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
        







