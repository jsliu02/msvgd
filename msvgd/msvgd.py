import numpy as np
import torch
from tqdm.notebook import trange

class MSVGD():
    def __init__(self, score=None, logdensity=None, density=None):
        '''
        Define either the score function, log-density, or density of the target distribution.
        Density may be up to multiplicative constant, log-density may be up to additive constant.
        '''
        self.score = lambda x: score(x.reshape(-1,1)).flatten() if score is not None else None
        self.logdensity = lambda x: logdensity(x.reshape(-1,1)).flatten() if logdensity is not None else None
        self.density = lambda x: density(x.reshape(-1,1)).flatten() if density is not None else None

        # compute gradient functions
        if score is not None:
            grad = self.score
            self.gradient_score = torch.func.vmap(self.score, in_dims=0)
        else:
            def missing_grad(*x):
                raise ValueError("No target score supplied.")
            self.gradient_score = missing_grad
        if logdensity is not None:
            def grad(x):
                x.grad = None
                sum_logdens = torch.func.vmap(self.logdensity)(x).sum()
                sum_logdens.backward()
                return x.grad
            self.gradient_logdensity = grad
        else:
            def missing_grad(*x):
                raise ValueError("No target log-density supplied.")
            self.gradient_logdensity = missing_grad
        if density is not None:
            def grad(x):
                x.grad = None
                sum_logdens = torch.func.vmap(lambda x: self.density(x).log())(x).sum()
                sum_logdens.backward()
                return x.grad
            self.gradient_density = grad
        else:
            def missing_grad(*x):
                raise ValueError("No target density supplied.")
            self.gradient_density = missing_grad
        if (score is None) and (logdensity is None) and (density is None):
            raise ValueError("No target distribution supplied.")

    def svgd_kernel(self, particles, h=-1):
        '''
        Compute the SVGD kernel.
        '''
        L2sq = torch.cdist(particles, particles)**2
        if h <= 0:
            h = L2sq.median() / self.logk
            
        Kxy = (-L2sq / h).exp()
        dxkxy = -Kxy @ particles
        sumkxy = Kxy.sum(axis=1).reshape(-1, 1)
        dxkxy += particles * sumkxy.tile(1, particles.shape[1])
        dxkxy *= 2/h

        return Kxy, dxkxy

    def mitotic_split(self, opt, grad):
        '''
        Perform mitotic split for mSVGD.
        '''
        old_particles = self.particles.clone()
        self.particles.grad = grad
        opt.step()
        self.particles = torch.concat([old_particles.detach(), self.particles.detach()], axis=0)

    def solve(self, x0, optimizer=torch.optim.Adam, optimizer_kwargs={"lr":1e-2}, method="score",
              max_iter=10_000, mitosis_splits=0, atol=1e-2, rtol=1e-8, bandwidth=-1, monitor_convergence=False,
              device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32,):
        '''
        Solve mSVGD optimization.
        '''
        match method:
            case "score":
                self.gradient = self.gradient_score
                mode = "manual"
            case "logdensity":
                self.gradient = self.gradient_logdensity
                mode = "autograd"
            case "density":
                self.gradient = self.gradient_density
                mode = "autograd"
                
        if torch.is_tensor(x0):
            self.particles = x0.clone()
        if not torch.is_tensor(x0):
            self.particles = torch.tensor(x0)
        self.particles = self.particles.to(device, dtype)
        
        for i in range(mitosis_splits+1):
            self.k = self.particles.shape[0]
            self.logk = np.log(self.k)
            self.MAP = (self.k == 1)
            
            if mode == "autograd":
                self.particles.requires_grad_()
            
            opt = optimizer(params=[self.particles], **optimizer_kwargs)

            with trange(max_iter) as pbar:
                for iteration in range(max_iter):
                    # compute gradient
                    grad_particles = -self.gradient(self.particles)
                    if not self.MAP:
                        kxy, dxkxy = self.svgd_kernel(self.particles, h=bandwidth)
                        grad_particles = (kxy @ grad_particles - dxkxy) / self.k
                        
                    # monitor gradient magnitude
                    if monitor_convergence and iteration % monitor_convergence == 0:
                        m = grad_particles.abs().max()
                        if monitor_convergence:
                            print(f'Iteration {iteration}, Max Grad = {m:.5f}')

                    # check convergence
                    if torch.all(torch.abs(grad_particles) <= atol + rtol * torch.abs(self.particles)):
                        pbar.update()
                        break
                    # update particles
                    else:
                        self.particles.grad = grad_particles
                        opt.step()
                        pbar.update()
                        
                # report completion status 
                m = grad_particles.abs().max()
                pbar.set_description(f'Split {i} finished with max grad = {m:.5f}')
                
            if i < mitosis_splits:
                self.mitotic_split(opt, grad_particles)