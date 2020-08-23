import torch
import torch.nn as nn
import numpy as np

class BaseRotationMatrix(nn.Module):
    def __init__(self,theta,train=True):
        super().__init__()
        if theta:
            self.theta = torch.Tensor([theta]).view(1)
        else:
            self.theta = torch.zeros(1)
        self.train = train
        self.R = self.construct_R(self.theta)
        
    def forward(self,x):
        if self.train:
            R = self.construct_R(self.theta)
            return R@x.view(-1)
        return self.R@x.view(-1)
    
    def coodinate_transform(self,basis):
        return basis@self.R.T
    
    def construct_R(self,**argv):
        raise Notimplementederror


class Rx(BaseRotationMatrix):
    def __init__(self,theta=None,train=True):
        super().__init__(theta,train)
        
    def construct_R(self,theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        return  torch.Tensor([[1 ,0 ,0],
                               [0 ,c  ,s],
                               [0 ,-s ,c]])

class Ry(BaseRotationMatrix):
    def __init__(self,theta=None,train=True):
        super().__init__(theta,train)
        
        
    def construct_R(self,theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        return  torch.Tensor([[c ,0 ,-s],
                               [0 ,1  ,0],
                               [s ,0 ,c]])


class Rz(BaseRotationMatrix):
    def __init__(self,theta=None,train=True):
        super().__init__(theta,train)
        
        
    def construct_R(self,theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        return  torch.Tensor([[c ,s ,0.],
                               [-s ,c  ,0],
                               [0. ,0 ,1]])
        
    
class EulerRotationXYZ(nn.Module):
    def __init__(self,yaw=None,roll=None,pitch=None,train=True):
        super().__init__()
        self._Rx = Rx(roll,train)
        self._Ry = Ry(pitch,train)
        self._Rz = Rz(yaw,train)
        
    def forward(self,x):
        return self._Rx(self._Ry(self._Rz(x)))
    
    def coodinate_transform(self,basis):
        basis = self._Rx.coodinate_transform(basis)
        basis = self._Ry.coodinate_transform(basis)
        basis = self._Rz.coodinate_transform(basis)
        return basis

    def construc_kinematic_matrix(self,yaw,roll,pitch):
        c_p = torch.cos(pitch)
        t_p = torch.tan(pitch)
        s_r = torch.sin(roll)
        c_r = torch.cos(roll)

        return  torch.Tensor([[1 ,s_r*t_p ,c_r*t_p],
                               [0 ,c_r  ,-s_r],
                               [0. ,s_r/c_p ,c_r/c_p]])
        
    def kinematic_transform(self,gyro_s,yaw,roll,pitch):
        M = self.construc_kinematic_matrix(yaw,roll,pitch)
        return M@gyro_s
