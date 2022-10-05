from turtle import forward
import torch
import copy
from collections.abc import Iterable
from numbers import Number

class GridApprox():
    def __init__(self, funtensor, histapprox=False):
        self.parent = funtensor
        self.histapprox = histapprox

    def __getitem__(self, x):
        x = [slice(j, j+1, 1.) if not isinstance(j, slice) else j for j in ((x,) if not isinstance(x, tuple) else x)]
        points_grid = [
            torch.arange(j.start, j.stop, j.step if j.step is not None else 1.)
            for j in x
        ]
        points_cart = torch.cartesian_prod(*points_grid)
        result = self.parent.forward(points_cart).reshape(*[len(j) for j in points_grid])
        if self.histapprox:
            spaces_grid = [
                torch.ones(len(j))*(x[i].step if x[i].step is not None else 1.)
                for i,j in enumerate(points_grid)
            ]
            spaces_cart = torch.prod(torch.cartesian_prod(*spaces_grid), -1)
            hist_result = result * spaces_cart.reshape(*[len(j) for j in spaces_grid])
            return hist_result
        return(result)

class FT(torch.nn.Module):
    
    def __init__(self, dtype=torch.float32, multiplication_op=lambda x,y: x*y, division_op=lambda x,y: x/y, addition_op=lambda x,y: x+y, subtraction_op=lambda x,y: x-y):
        super(FT, self).__init__()
        self.dtype = dtype
        self.multiplication_op = multiplication_op
        self.division_op = division_op
        self.addition_op = addition_op
        self.subtraction_op = subtraction_op
        self.view = ''

    @staticmethod
    def from_torch_module(module):
        c = FT()
        c.module = (module,)
        c.parameters = module.parameters
        return c

    @property
    def infer(self):
        c = copy.copy(self)
        for param in c.parameters():
            param.requires_grad = True 
        return c

    @property
    def train(self):
        c = copy.copy(self)
        for param in c.parameters():
            param.requires_grad = True 
        return c

    def to_tensor_sum(self, f, t, view, force_train=[], force_infer=[]):
        c = self.v('_x' + view[0]).where(x=f)
        for i in range(f+1, t):
            d = self.v('_x' + view[i]).where(x=i)
            if i in force_train:
                d = d.train
            if i in force_infer:
                d = d.infer
            c += d
        return c

    def v(self, key):
        if key[0] == '_' and key.count('_') == 1:
            c = copy.copy(self)
            c.view = key[1:]
            c.view = self.__compute_out_index(c.view, "")
            return c
        else:
            return self.__getattribute__(key)
        
    def __getattr__(self, key):
        return self.v(key)
        
    def __typecast(self, o, view=None):
        if isinstance(o, Number):
            r = FT()
            def forward(x):
                return torch.ones((x.shape[0])) * o
            r.forward = forward
            return r
        if isinstance(o, FT):
            return o
        
    def __compute_out_index(self, i1, i2, upper=True):
        out = list(sorted(set(i1+i2), key=lambda x: x.lower())) # TODO sorted?
        return "".join([(j.upper() if upper else j) if (i1.count(j)+i2.count(j))%2 == 0 else j for j in out])
    
    def __compute_permutation(self, l_view, r_view, out_view):
        l_perm = []
        r_perm = []
        for i,v in enumerate(out_view):
            if v.lower() in l_view: l_perm.append(i)
            if v.lower() in r_view: r_perm.append(i)
        return tuple(l_perm), tuple(r_perm)
    
    def __rmul__(self, o):
        o = self.__typecast(o, self.view)
        c = copy.copy(self)
        c.view = self.__compute_out_index(self.view, o.view)
        c.old_forward = c.forward
        def forward(x):
            l_perm, r_perm = self.__compute_permutation(self.view, o.view, c.view)
            l = self.forward(x[:, l_perm])
            r = o.forward(x[:, r_perm])
            return self.multiplication_op(l, r)
        c.forward = forward
        return c
    
    def __lmul__(self, o):
        return self.__rmul__(o)
    
    def __mul__(self, o):
        return self.__rmul__(o)
    
    def __add__(self, o):
        o = self.__typecast(o, self.view)
        c = copy.copy(self)
        c.view = self.__compute_out_index(self.view, o.view, upper=False)
        c.old_forward = c.forward
        def forward(x):
            l_perm, r_perm = self.__compute_permutation(self.view, o.view, c.view)
            return self.addition_op(self.forward(x[:, l_perm]), o.forward(x[:, r_perm]))
        c.forward = forward
        return c

    def __sub__(self, o):
        return self + (-1)*o

    def __truediv__(self, o):
        return self * (1/o)

    def where(self, **kwargs):
        c = copy.copy(self)
        view = list(self.view)
        for k,_ in kwargs.items(): view.remove(k)
        c.view = "".join(view)
        c.old_forward = c.forward
        def forward(x):
            z = torch.zeros((x.shape[0], len(self.view)))
            for k,v in kwargs.items(): z[:, self.view.index(k)] = v
            for i in range(x.shape[1]): z[:, self.view.index(c.view[i])] = x[:, i]
            return self.forward(z)
        c.forward = forward
        return c

    def comp(self, o):
        c = copy.copy(self)
        c.old_forward = c.forward
        def forward(x):
            return self.forward(o.forward(x))
        c.forward = forward
        c.view = o.view
        return c

    @property
    def gridapprox(self):
        return GridApprox(self)

    @property
    def histapprox(self):
        return GridApprox(self, histapprox=True)

    @property
    def scalar(self):
        return self(torch.tensor([[]]))

    def gd(self, epochs=10, optimizer=torch.optim.Adam, lr=None):
        optimizer = optimizer(self.parameters())
        loss = self.scalar
        

    def histconv(self, index, conv_from, conv_to, bins):
        c = copy.copy(self)
        i = index.upper()
        ii = self.view.index(i)
        new_view = c.view.replace(index.upper(), '')
        c.old_forward = self.forward
        def forward(x):
            if self.view == '':
                post = torch.tensor
            else:
                post = torch.cat
            linspace = torch.linspace(conv_from, conv_to, bins)
            batch = torch.cat(
                [x[:, :ii],
                torch.ones(len(x)).reshape(-1, 1), 
                x[:, ii:]],
            1).repeat((1, len(linspace))).reshape(len(x) * len(linspace), -1)
            linspace_repeated = linspace.repeat(len(x))
            batch[:, ii] = linspace_repeated
            y_batch = c.old_forward(batch) 
            y_batch = y_batch * ((conv_to - conv_from) / bins)
            y_batch = y_batch.reshape(len(x), len(linspace))
            result = y_batch.sum(1)
            return result
        c.forward = forward
        c.view = new_view
        return c

    def boxconv(self, f, t, bins=None):
        if bins is None: bins = t-f
        c = self
        for i in [j for j in self.view if j.isupper()]:
            c = c.histconv(i, f, t, bins)
        return c

    def bc(self, f, t, bins=None):
        return self.boxconv(f, t, bins)

    def forward(self, x):
        return self.module[0].forward(x)

class Gaussian(FT):
    def __init__(self, mu, cov):
        super(Gaussian, self).__init__()
        self.distribution = torch.distributions.MultivariateNormal(
            torch.tensor(mu), 
            torch.tensor(cov)
        )

    def forward(self, x):
        return torch.exp(self.distribution.log_prob(x))

class Uniform(FT):
    def __init__(self, low, high):
        super(Uniform, self).__init__()
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.supp = torch.prod(self.high - self.low)

    def forward(self, x):
        return self.supp * (torch.all(self.high[None, :] > x, 1) & torch.all(self.low[None, :] < x, 1))



