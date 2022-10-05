from funtensors import FT
import torch

def test_mul_type():
    assert(isinstance(FT()*FT(), FT))

def test_add_type():
    assert(isinstance(FT()+FT(), FT))

def test_div_type():
    assert(isinstance(FT()/FT(), FT))

def test_self_convolution_view():
    class TestFt(FT):
        def forward(self, x):
            return 1.
    x = TestFt()
    assert(x.view == '')
    assert(x._ijk.view == 'ijk')
    assert(x._ii.view == 'I')
    assert(x._ijki.view == 'Ijk')

def test_mul_convolution_view():
    class TestFt(FT):
        def forward(self, x):
            return torch.ones((x.shape[0], 1))
    x = TestFt()
    y = TestFt()
    assert(x._ij.view == 'ij')
    assert(y._jk.view == 'jk')
    assert((x._ij*y._jk).view == 'iJk')

def test_mul_convolution_result():
    class TestFt(FT):
        def forward(self, x):
            return torch.ones((x.shape[0], 1))
    x = TestFt()
    y = TestFt()
    z = x._ij*y._jk
    assert (z.view == 'iJk')
    r = z(
        torch.FloatTensor(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 0],
                [1, 0, 1]
            ]
        )
    )
    assert(r[0] == 1.)
    assert(r[1] == 1.)
    assert(r[2] == 1.)
    assert(r[3] == 1.)

def test_getitem_zero():
    x = FT(rank=1)
    y = x[5:6]
    assert(y[4] == 0)
    assert(y[7] == 0)
    z = FT(rank=2)[5:7, 2:3]
    assert(z[8, 1] == 0)
    assert(z[1, 8] == 0)

def test_gridapprox_1d():
    class TestFt(FT):
        def forward(self, x):
            return torch.ones((x.shape[0], 1))
    x = TestFt()
    assert (
        torch.all(
            x.gridapprox[5:8:0.5] == torch.tensor(
                [
                    1.,
                    1.,
                    1.,
                    1.,
                    1.,
                    1.
                ]
            )
        )
    )
    assert (
        torch.all(
            x.gridapprox[5:8:1] == torch.tensor(
                [
                    1.,
                    1.,
                    1.
                ]
            )
        )
    )

def test_gridapprox_2d():
    class TestFt(FT):
        def forward(self, x):
            return torch.sum(x, -1)
    x = TestFt()
    assert (
        torch.all(
            x.gridapprox[5:7, 6:8] == torch.tensor(
                [
                    [5.+6., 5.+7.],
                    [6.+6., 6.+7.]
                ]
            )
        ).item()
    )

def test_histapprox_2d():
    class TestFt(FT):
        def forward(self, x):
            return torch.sum(x, -1)
    x = TestFt()
    assert (
        torch.all(
            x.histapprox[5:7:0.5, 6:8:0.5] == torch.tensor(
                [
                    [2.7500, 2.8750, 3.0000, 3.1250],
                    [2.8750, 3.0000, 3.1250, 3.2500],
                    [3.0000, 3.1250, 3.2500, 3.3750],
                    [3.1250, 3.2500, 3.3750, 3.5000]
                ]
            )
        ).item()
    )

"""
def test_histconv():
    class ThreeTestFt(FT):
        def forward(self, x):
            return 3*torch.ones((x.shape[0], 1))
    class FourTestFt(FT):
        def forward(self, x):
            return 4*torch.ones((x.shape[0], 1))
    a = ThreeTestFt()
    b = FourTestFt()
    c = (a._ij * b._jk).histconv('j', -1, 1, 100)
    assert (
        torch.all(
            c.gridapprox[-1:1:0.5, -1:1:0.5] == torch.tensor(
                [
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.]
                ]
            )
        ).item()
    )

def test_convall():
    class ThreeTestFt(FT):
        def forward(self, x):
            return 3*torch.ones((x.shape[0], 1))
    class FourTestFt(FT):
        def forward(self, x):
            return 4*torch.ones((x.shape[0], 1))
    a = ThreeTestFt()
    b = FourTestFt()
    c = (a._ij * b._jk).boxconv(-1, 1, 4)
    assert (
        torch.all(
            c.gridapprox[-1:1:0.5, -1:1:0.5] == torch.tensor(
                [
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.]
                ]
            )
        ).item()
    )

def test_convuniform():
    from funtensors import Gaussian, Uniform
    x = Uniform([0.2, 0.3], [0.6, 0.8])
    y = Uniform([0.3, 0.3], [0.8, 0.8])
    (x._ij * y._jk).boxconv(-2., 2., 4).gridapprox[-2:2, -2:2]
    assert False

def test_histconv():
    class ThreeTestFt(FT):
        def forward(self, x):
            return 3*torch.ones((x.shape[0], 1))
    class FourTestFt(FT):
        def forward(self, x):
            return 4*torch.ones((x.shape[0], 1))
    a = ThreeTestFt()
    b = FourTestFt()
    c = (a._ij * b._jk).histconv('j', -1, 1, 3)
    assert (
        torch.all(
            c.gridapprox[-1:2:1, -1:2:1] == torch.tensor(
                [
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.]
                ]
            )
        ).item()
    )
"""

def test_bc():
    class TestFt(FT):
        def forward(self, x):
            y = torch.tensor(
                [
                    [0, 1],
                    [3, 4]
                ]
            )
            r = [y[z[0].long(), z[1].long()] for z in x]
            return torch.tensor(r)
    a = TestFt()
    b = TestFt()
    c = a._ij * b._jk
    c = c.histconv('j', 0, 1, 2)
    print(
        c(
            torch.tensor(
                [
                    [1, 1],
                    [0, 1]
                ]
            )
        )
    )
    assert False
    