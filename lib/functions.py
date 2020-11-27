import torch
from abc import abstractmethod, ABC



class Func(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, x, *kwargs):
        ...

    @abstractmethod
    def dx(self, x, **kwargs):
        """
        gradient
        """
        ...


class Drift_linear(Func):

    def __init__(self, L, M):
        """
        Parameters
        ----------
        L: torch.Tensor
            Tensor of shape (d,d)
        M: torch.Tensor
            Tensor of shape (d,d)
        """
        self.L=L
        self.M=M

    def __call__(self, t, x, a):
        """Returns Lx + Ma
        Parameters
        ----------
        t: torch.Tensor
            time. Tensor of shape (batch_size, 1)
        x: torch.Tensor
            State. Tensor of shape (batch_size, dim)
        a: torch.Tensor
            Action. Tensor of shape (batch_size, dim)
        
        Returns
        -------
        drift: torch.Tensor
            Tensor of shape (batch_size, dim)
        
        """
        Lx = torch.matmul(self.L, x.unsqueeze(2)) # (batch_size, d, 1)
        Ma = torch.matmul(self.M, a.unsqueeze(2)) # (batch_size, d, 1)
        return (Lx+Ma).squeeze(2) # (batch_size, d)

    def dx(self, t, x, a):
        pass


class Diffusion_constant(Func):
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, t, x, a):
        return torch.ones_like(x)*self.sigma # (batch_size, d)

    def dx(self, t, x, a):
        pass


class Hamiltonian(Func):

    def __init__(self, drift: Func, diffusion: Func, running_cost: Func):

        self.drift = drift
        self.diffusion = diffusion
        self.running_cost = running_cost

    def __call__(self, t, x, a, y, z):
        """
        Parameters
        ----------
        t: torch.Tensor
            time. Tensor of shape (batch_size, 1)
        x: torch.Tensor
            State. Tensor of shape (batch_size, dim)
        a: torch.Tensor
            Action. Tensor of shape (batch_size, dim)
        y: torch.Tensor
            Adjoint state. Tensor of shape (batch_size, dim)
        z: torch.Tensor
            Diffusion in adjoint process. Tensor of shape (batch_size, dim, dim)
        Note
        ----
        I'm considering the diffuion of the SDE to be diagonal
        """
        diffusion_z = torch.bmm(self.diffusion(t,x,a).unsqueeze(1), z) # (batch_size,1,d). I'm considering the diffusion to be diagonal!
        trace_diffusion_z = torch.sum(diffusion_z, 2) # (batch_size, 1) 
        H = torch.sum(self.drift(t,x,a)*y,1,keepdim=True) + trace_diffusion_z + self.running_cost(x,a) 
        return H #(batch_size, 1)
    
    
    def dx(self, t, x, a, y, z, create_graph, retain_graph):
        x.requires_grad_(True)
        H = self.__call__(t,x,a,y,z)
        dHdx = torch.autograd.grad(H,x,grad_outputs=torch.ones_like(H),only_inputs=True, create_graph=create_graph, retain_graph=retain_graph)[0]
        return dHdx # (batch_size, d)


class QuadraticFinalCost(Func):

    def __init__(self, R):
        """
        Parameters
        ----------
        R: torch.Tensor
            Tensor of shape (d,d)
        """
        self.R = R

    def __call__(self, x):
        """
        Calculcate x.T * R * x, where x is a batch of data. 
        I vectorize all the calculations
        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape (batch_size, d)
        Returns
        -------
        xMx: torch.Tensor
            Tensor of shape batch_size, 1
        """
        Rx = torch.matmul(self.R, x.unsqueeze(2)) # (batch_size, d, 1)
        xRx = torch.bmm(x.unsqueeze(1),Rx) # (batch_size, 1, 1)
        return xRx.squeeze(2) # (batch_size, 1)
    
    def dx(self, x, create_graph, retain_graph):
        x.requires_grad_(True)
        g = self.__call__(x)
        dgdx = torch.autograd.grad(g,x,grad_outputs=torch.ones_like(g), only_inputs=True, create_graph=create_graph, retain_graph=retain_graph)[0]
        return dgdx


class QuadraticRunningCost(Func):

    def __init__(self, C, D, F):
        """
        Parameters
        ----------
        C: torch.Tensor
            Tensor of shape (d,d)
        D: torch.Tensor
            Tensor of shape (d,d)
        F: torch.Tensor
            Tensor of shape (d,d)
        """
        self.C = C
        self.D = D
        self.F = F

    def __call__(self, x, a):
        """
        Calculate x.T*C*x + a.T*D*a + 2*x.T*F*a
        Parameters
        ----------
        x: torch.Tensor
            tensor of shape (batch_size, d)
        a: torch.Tensor
            tensor of shape (batch_size, d)
        """
        Cx = torch.matmul(self.C, x.unsqueeze(2)) # (batch_size, d, 1)
        xCx = torch.bmm(x.unsqueeze(1),Cx) # (batch_size, 1, 1)
        Da = torch.matmul(self.D, a.unsqueeze(2)) # (batch_size, d, 1)
        aDa = torch.bmm(a.unsqueeze(1),Da) # (batch_size, 1, 1)
        Fa = torch.matmul(self.F, a.unsqueeze(2)) # (batch_size, d, 1)
        xFa = torch.bmm(x.unsqueeze(1),Fa) # (batch_size, 1, 1)
        return (xCx + aDa + 2*xFa).squeeze(2)

    def dx(self,x):
        pass

