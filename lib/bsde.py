import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod

from lib.networks import FFN
from lib.functions import Func, Hamiltonian


class FBSDE(nn.Module):

    def __init__(self, d: int, ffn_hidden: List[int], drift: Func, diffusion: Func, running_cost: Func, final_cost: Func):
        """
        Initialisation of the FBSDE that we want to solve.
        
        Paramters
        ---------
        d: int
            dim of the process
        ffn_hidden: List[int]
            hidden sizes of the Feedforward networks that parametrise the policy, Y, Z
        drift: Func
            Drift of the controlled process. 
        diffusion: Func
            Diffusion of the controlled process
        running_cost: Func
            Running cost of the control
        final_cost: Func
            Final cost of the control
        """
        super().__init__()
        self.d = d

        self.drift = drift
        self.diffusion = diffusion
        
        self.running_cost = running_cost
        self.final_cost = final_cost
        
        self.alpha = FFN(sizes = [d+1] + ffn_hidden + [d]) # +1 is for time
        self.Y = FFN(sizes = [d+1] + ffn_hidden + [d]) # Adjoint state. +1 is for time
        self.Z = FFN(sizes = [d+1] + ffn_hidden + [d*d]) # Diffusion of adjoint BSDE. It takes values in R^{d\times d}
        self.H = Hamiltonian(drift = self.drift, diffusion=self.diffusion, running_cost=self.running_cost)

    
    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        Note
        ----
        I am assuming uncorrelated Brownian motion
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            current_t = t*torch.ones(batch_size, 1, device=device)
            tx = torch.cat([current_t, x[:,-1,:]],1)
            a = self.alpha(tx)
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            
            x_new = x[:,-1,:] + self.drift(current_t, x[:,-1,:], a)*h + self.diffusion(current_t, x[:,-1,:], a)*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments
    
    def bsdeint(self, ts: torch.Tensor, x0: torch.Tensor): 
        """
        Local errors at each timestep in timegrid ts to solve the BSDE
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        
        """
        with torch.no_grad():
            x, brownian_increments = self.sdeint(ts, x0)
        
        device=x.device
        final_value = self.final_cost.dx(x[:,-1,:], create_graph=False, retain_graph=False) # (batch_size, d)
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        
        Y = self.Y(tx) # (batch_size, L, 2)
        Z = self.Z(tx).view(batch_size, len(ts), self.d, self.d) # (batch_size, L, d, d)
        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts):
            if t==ts[-1]:
                pred = Y[:,idx,:]
                target = final_value
            else:
                h = ts[idx+1] - ts[idx]
                current_t = t*torch.ones(batch_size, 1, device=device)
                y = Y[:,idx,:]
                tx = torch.cat([current_t, x[:,idx,:]],1)
                with torch.no_grad():
                    a = self.alpha(tx)
                z = Z[:,idx,:]
                #stoch_int = torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
                stoch_int = torch.bmm(Z[:,idx,...], brownian_increments[:,idx,:].unsqueeze(2)).squeeze(2) # (batch_size, d)
                dHdx = self.H.dx(t=current_t, 
                        x=x[:,idx,:],
                        a=a,
                        y=y,
                        z=z,
                        create_graph=True,
                        retain_graph=True)
                pred = y - dHdx*h + stoch_int # euler timestep
                target = Y[:,idx+1,:].detach()
            loss += loss_fn(pred, target)
        return loss

    
    def loss_policy(self, ts: torch.Tensor, x0: torch.Tensor):
        """ 
        We want to find
        - argmin_{a} H(t,X,Y,Z,a) for all t
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        """
        with torch.no_grad():
            x, brownian_increments = self.sdeint(ts, x0)
        batch_size = x.shape[0]
        device=x.device
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        with torch.no_grad():
            Y = self.Y(tx) # (batch_size, L, d)
            Z = self.Z(tx).view(batch_size, len(ts), self.d, self.d) # (batch_size, L, d, d)
        
        loss = 0
        for idx, t in enumerate(ts):
            current_t = t*torch.ones(batch_size, 1, device=device)
            y = Y[:,idx,:]
            tx = torch.cat([current_t, x[:,idx,:]],1)
            a = self.alpha(tx)
            z = Z[:,idx,:]
            H = self.H(t=current_t, x=x[:,idx,:],a=a,y=y,z=z)
            loss += H
        return loss.mean()

            
            




