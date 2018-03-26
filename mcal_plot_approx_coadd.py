import numpy
from numpy import array
import biggles
import ngmix

units=1.0e-3

class Fitter(object):
    def __init__(self, shear, m, merr):
        self.shear=shear
        self.m=m
        self.merr=merr
    
    def go(self):
        # [m, alpha]
        guess = numpy.array([0.0001, 1.0])

        result = ngmix.fitting.run_leastsq(
            self._calc_fdiff,
            guess,
            0,
        )
        self.result=result

    def __call__(self, shear):
        m, alpha = self.result['pars']
        return m + alpha * shear**2

    def _calc_fdiff(self, pars):
        predict = pars[0] + pars[1] * self.shear**2

        ydiff = (predict - self.m)/self.merr

        return ydiff

def print_res(res):
    mess="%(type)s m: %(m)g +/- %(m_err)gg alpha: %(alpha)g +/- %(alpha_err)g"
    print(mess % res)


def make_points(shears, biases, bias_errors, color, res):
    pts=biggles.Points(shears,biases, type='filled diamond', color=color,size=2.5)
    ptsc=biggles.Points(shears,biases, type='diamond', color='black',size=2.5)
    ptserr=biggles.SymmetricErrorBarsY(simshears,simbiases,simerrs, color=simcolor)


    if res is not None:
        xp = numpy.linspace(0, shears[-1])
        fitc = biggles.Curve(
            xp,
            res['m'] + res['alpha']/units*xp**2,
            color=color,
        )

        fitc.label=r'$%.1f \gamma^2$' % res['alpha']
    else:
        fitc=None

    return pts, ptsc, ptserr, fitc

def run_fitter(shears, biases, bias_errors):
    fitter=Fitter(shears, biases, bias_errors)
    fitter.go()
    m,alpha = fitter.result['pars']
    m_err, alpha_err = numpy.sqrt( numpy.diag(fitter.result['pars_cov']))
    res={
        'm':m,
        'm_err':m_err,
        'alpha':alpha,
        'alpha_err':alpha_err,
    }
    return res


fname='results-noise0-m0.6-ring.txt'
biggles.configure('default','fontsize_min',2.5)
m,merr=numpy.loadtxt(fname, delimiter=' ', unpack=True)

merr += 1.0e-5

shears=array( [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])

xp = numpy.linspace(0, shears[-1])

plt=biggles.FramedPlot()
plt.aspect_ratio=1.0/1.618
plt.xlabel=r'$\gamma_{true}$'
plt.ylabel=r'$m [10^{-3}]$'
#plt.xrange=[-0.0049, 0.0849]
plt.xrange=[0, 0.11]


#simcolor='seagreen'
simcolor='steelblue'
nccolor='seagreen'
#simcolor='firebrick2'
#simcolor='maroon'


simshears=array([0.02,0.06,0.08,0.10])
simbiases=array([
    5.570e-04/units,
    2.386e-03/units,
    8.202e-03/units,
    1.371e-02/units, 
])
simerrs=array([
    0.00036/units,
    0.00136/units,
    0.00153/units,
    0.001844/units,   
])


simres = run_fitter(simshears, simbiases*units, simerrs*units)
simpts, simptsc, simerr, fitc = make_points(simshears, simbiases, simerrs, simcolor, simres)
simpts.label='coadd'
simres['type']='coadd'
print_res(simres)


ncshears=array([0.08,0.10])
ncbiases=array([
    7.485e-03/units,
    1.287e-02/units, 
])
ncerrs=array([
    0.000874/units,
    0.000906/units,   
])
ncres=None

ncpts, ncptsc, ncerr, ncfitc = make_points(
    ncshears,
    ncbiases,
    ncerrs,
    nccolor,
    ncres,
)
ncpts.label='no coadd'
#ncres = run_fitter(ncshears, ncbiases*units, ncerrs*units)
#ncres['type']='no coadd'
#print_res(ncres)


key = biggles.PlotKey(
    0.1,0.9,
    [simpts,fitc, ncpts],
    halign='left',
)


ax = numpy.array([-0.1,0.2])
z = biggles.Curve(ax,ax*0)

allowed=biggles.FillBetween(ax, [-1.0e-3/units]*2,
                            ax, [1.0e-3/units]*2, color='gray90')


plt.add(
    allowed,
    z,

    simpts,
    simerr,
    simptsc,
    fitc,

    ncpts,
    ncptsc,
    ncerr,

    key,
)

plt.write_eps('results-noise0-m0.6-coadd.eps')
