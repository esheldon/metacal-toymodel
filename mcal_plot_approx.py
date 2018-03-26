import numpy
from numpy import array
import biggles
import ngmix

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

fname='results-noise0-m0.6-ring.txt'
biggles.configure('default','fontsize_min',2.5)
m,merr=numpy.loadtxt(fname, delimiter=' ', unpack=True)

merr += 1.0e-5

shears=array( [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])

fitter=Fitter(shears, m, merr)
fitter.go()
fm,falpha = fitter.result['pars']
fmerr, falphaerr = numpy.sqrt( numpy.diag(fitter.result['pars_cov']))

#coeffs=numpy.polyfit(shears, m, 2)
#print("coeffs:",coeffs)

print("m + alpha g^2")
print("m: %g +/- %g alpha: %g +/- %g" % (fm,fmerr,falpha,falphaerr))
#ply=numpy.poly1d(coeffs)

xp = numpy.linspace(0, shears[-1])

plt=biggles.FramedPlot()
plt.aspect_ratio=1.0/1.618
plt.xlabel=r'$\gamma_{true}$'
plt.ylabel=r'$m [10^{-3}]$'
#plt.xrange=[-0.0049, 0.0849]
plt.xrange=[0, 0.11]


mcolor='firebrick2'
#mcolor='steelblue'
#mcolor='maroon'

units=1.0e-3
pts = biggles.Points(
    shears, m/units, type='filled circle', color=mcolor, size=2.5,
)
ptsc = biggles.Points(
    shears, m/units, type='circle', color='black', size=2.5,
)

pts.label='toy model'

#simcolor='seagreen'
simcolor='steelblue'
#simcolor='firebrick2'
#simcolor='maroon'


#simpts=biggles.Points([0.02], [0.03], type='filled diamond', color=simcolor,size=2.5)
#simptsc=biggles.Points([0.02], [0.03], type='diamond', color='black',size=2.5)
#simerr=biggles.SymmetricErrorBarsY([0.02], [0.03], [0.31], color=simcolor)



simshears=array([0.02,0.06,0.10])
simbiases=array([
    0.03,
    2.064e-3/units,
    0.00581/units],
)
simerrs=array([
    0.31,
    3.210e-04/units,
    0.000280/units ],
)
simpts=biggles.Points(simshears,simbiases, type='filled diamond', color=simcolor,size=2.5)
simptsc=biggles.Points(simshears,simbiases, type='diamond', color='black',size=2.5)
simerr=biggles.SymmetricErrorBarsY(simshears,simbiases,simerrs, color=simcolor)
#simerr2=biggles.SymmetricErrorBarsY([0.02+0.00005], [0.03+0.03], [0.31], color='black')

simfitter=Fitter(simshears, simbiases*units, simerrs*units)
simfitter.go()
sim_m,sim_alpha = simfitter.result['pars']
sim_merr, sim_alphaerr = numpy.sqrt( numpy.diag(fitter.result['pars_cov']))

print("sim m + alpha g^2")
print("sim m: %g +/- %g alpha: %g +/- %g" % (sim_m,sim_merr,sim_alpha,sim_alphaerr))


simpts.label='image simulation'

#perr = biggles.SymmetricErrorBarsY(shears, m, merr)

#c = biggles.Curve(xp,fitter(xp),color='blue')
c = biggles.Curve(xp,xp**2/units,color=mcolor, type='shortdashed')
#c.label=r'$%.2f + %.2f*g^2$' % (fm, falpha)
c.label=r'$1.0 \gamma^2$'

simc = biggles.Curve(xp,sim_m + sim_alpha/units*xp**2,color=simcolor)
#c.label=r'$%.2f + %.2f*g^2$' % (fm, falpha)
simc.label=r'$%.1f \gamma^2$' % sim_alpha



key = biggles.PlotKey(0.1,0.9,[pts,c,simpts,simc], halign='left')


ax = numpy.array([-0.1,0.2])
z = biggles.Curve(ax,ax*0)

allowed=biggles.FillBetween(ax, [-1.0e-3/units]*2,
                            ax, [1.0e-3/units]*2, color='gray90')


plt.add(allowed, z, c, simc, pts, ptsc, simpts, simerr, simptsc, key)

#plt.show()

plt.write_eps('results-noise0-m0.6.eps')
