"""NumPy directional derivatives through mixed second order."""
import numpy as np


class Jet:
    """Value, directional derivatives u and v, and mixed derivative uv."""
    def __init__(self, x, u=0., v=0., uv=0.):
        self.x = np.asarray(x, dtype=float)
        self.u, self.v, self.uv = [np.broadcast_to(a, self.x.shape)
                                   for a in (u, v, uv)]

    def parts(self):
        return self.x, self.u, self.v, self.uv

    def linear(self, op):
        return Jet(*(op(p) for p in self.parts()))

    def __getitem__(self, idx):
        return self.linear(lambda p: p[idx])

    def __add__(self, other):
        b = jet(other)
        return Jet(*(a + c for a, c in zip(self.parts(), b.parts())))

    __radd__ = __add__

    def __neg__(self):
        return self.linear(lambda p: -p)

    def __sub__(self, other):
        return self + (-jet(other))

    def __rsub__(self, other):
        return jet(other) + (-self)

    def __mul__(self, other):
        return bilinear(self, jet(other), np.multiply)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return bilinear(self, jet(other), np.matmul)

    def unary(self, f, df, ddf):
        return Jet(f, df*self.u, df*self.v,
                   df*self.uv + ddf*self.u*self.v)

    def exp(self):
        f = np.exp(self.x)
        return self.unary(f, f, f)

    def power(self, p):
        if p == 0:
            return Jet(np.ones_like(self.x))
        if p == 1:
            return self
        x = self.x
        return self.unary(x**p, p*x**(p-1), p*(p-1)*x**(p-2))


def jet(a):
    return a if isinstance(a, Jet) else Jet(a)


def bilinear(a, b, op):
    return Jet(op(a.x, b.x),
               op(a.u, b.x) + op(a.x, b.u),
               op(a.v, b.x) + op(a.x, b.v),
               op(a.uv, b.x) + op(a.u, b.v) +
               op(a.v, b.u) + op(a.x, b.uv))


def join(items, concatenate=False):
    op = np.concatenate if concatenate else np.stack
    return Jet(*(op(a) for a in zip(*(j.parts() for j in items))))
