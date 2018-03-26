from __future__ import print_function
import sys
import numba
from numba import jit

import numpy
from numpy import sqrt
import time

import argparse

try:
    xrange=xrange
except:
    xrange=range

parser=argparse.ArgumentParser()

parser.add_argument('-n', type=int, default=10000, help='number of trials')
parser.add_argument('--s1', type=float, default=0.01, help='shear in component 1')
parser.add_argument('--noise', type=float, default=0.2, help='noise on shape')
parser.add_argument('--shapenoise', type=float, default=0.2, help='noise on shape')

parser.add_argument('--seed', type=int, default=None, help='seed for rng')


@jit(nopython=True, cache=True)
def shear_reduced(g1, g2, s1, s2):
        
    A = 1 + g1*s1 + g2*s2
    B = g2*s1 - g1*s2
    denom_inv = 1./(A*A + B*B)

    tg1 = A*(g1 + s1) + B*(g2 + s2)
    tg2 = A*(g2 + s2) - B*(g1 + s1)

    tg1 *= denom_inv
    tg2 *= denom_inv

    return tg1, tg2

@jit(nopython=True, cache=True)
def draw_shape(shapenoise):
    """
    truncated gaussian for shapes
    """
    while True:
        g1 = shapenoise*numpy.random.randn()
        g2 = shapenoise*numpy.random.randn()

        g=numpy.sqrt(g1**2 + g2**2)
        if g < 1.0:
            break

    return g1,g2

@jit(nopython=True, cache=True)
def add_err_trunc(g1, g2, noise):
    """
    truncated gaussian for errors
    """
        
    err1 = noise*numpy.random.randn()
    err2 = noise*numpy.random.randn()

    ng1 = g1 + err1
    ng2 = g2 + err2

    ng=numpy.sqrt(ng1**2 + ng2**2)

    if ng > 1.0:
        ng1 /= ng
        ng2 /= ng

    return ng1,ng2


@jit(nopython=True, cache=True)
def add_err_shear(g1, g2, err1, err2):
        

    tg1, tg2 = shear_reduced(g1, g2, err1, err2)
    return tg1, tg2


@jit(nopython=True, cache=True)
def rotate_shape(g1, g2, theta_radians):
    twotheta = 2.0*theta_radians

    cos2angle = numpy.cos(twotheta)
    sin2angle = numpy.sin(twotheta)
    g1rot =  g1*cos2angle + g2*sin2angle
    g2rot = -g1*sin2angle + g2*cos2angle

    return g1rot, g2rot

@jit(nopython=True, cache=True)
def go(n, shapenoise, s1, s2, noise, step):

    g1sum=0.0
    g2sum=0.0
    R1sum=0.0
    R1sqsum=0.0

    dgamma = 0.02

    theta = numpy.pi*0.5

    for i in xrange(n):

        g1o, g2o, = draw_shape(shapenoise)

        for ipair in xrange(2):
            if ipair == 1:
                g1, g2 = rotate_shape(g1o, g2o, theta)
            else:
                g1, g2 = g1o, g2o

            g1, g2 = shear_reduced(g1, g2, s1, s2)

            g1n, g2n = add_err_trunc(g1, g2, noise) 
            #g1n, g2n = add_err_shear(g1, g2, err1, err2) 

            # shear the shape with error
            g1n_1p, g2n_1p = shear_reduced(g1n, g2n,  step, 0.0)
            g1n_1m, g2n_1m = shear_reduced(g1n, g2n, -step, 0.0)

            R1 = (g1n_1p - g1n_1m)/dgamma

            g1sum   += g1n
            g2sum   += g2n

            R1sum += R1
            R1sqsum += R1**2


    nn = 2*n
    g1mean = g1sum/nn
    g2mean = g2sum/nn

    # for ring
    g1err = numpy.sqrt(noise/nn)
    g2err = numpy.sqrt(noise/nn)

    R1 = R1sum/nn
    R1var = R1sqsum/nn - R1**2
    R1err = numpy.sqrt(R1var/nn)

    return g1mean, g1err, g2mean, g2err, R1, R1err
 
def main():
    args = parser.parse_args()

    numpy.random.seed(args.seed)
    s2=0.0

    step=0.01

    print("calculating")
    tm=time.time()
    g1mean,g1err, g2mean,g2err,R1,R1err = go(
        args.n,
        args.shapenoise,
        args.s1,
        s2,
        args.noise,
        step)
    print(time.time()-tm)

    frac=g1mean/args.s1-1
    fracerr=g1err/args.s1

    corr1=g1mean/R1
    corr1err=g1err/R1
    corrfrac=corr1/args.s1-1
    corrfracerr=corr1err/args.s1

    print()
    print("errors are 2 sigma")
    print("g:        %g +/- %g" % (g1mean,g1err))
    print("R1:       %g +/- %g" % (R1,2*R1err))
    print("frac:     %g +/- %g" % (frac,2*fracerr))
    print("corrfrac: %g +/- %g" % (corrfrac,2*corrfracerr))


    return


if __name__=="__main__":
    main()
