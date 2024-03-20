import numpy as np
import random

# Functions based on EDGE1 subgrid physics, created by Eric Andersson.

class Parameters:
    def __init__(self,model):

        if model == 'EDGE1':
            self.SNIImmin = 8.0
            self.SNIImmax = 120.0

            self.OBmmin=8.0
            self.OBmmax=120.0

            self.AGBmmin=0.5
            self.AGBmmax=8.0

            self.SNIatmin = 38.0
        else:
            raise NotImplementedError("Model not implemented")

class Stars:
    def __init__(self,npartmax=1000):
        self.mass = np.zeros(npartmax)
        self.mass_birth = np.zeros(npartmax)
        self.age = np.zeros(npartmax)
        self.metal = np.zeros(npartmax)
        self.npar = 0
        self.time = 0
        self.with_SNII = True
        self.with_SNIa = True
        self.with_OB = True
        self.with_AGB = True

    def add_stars(self,mass,met):
        self.mass[self.npar] = mass
        self.mass_birth[self.npar] = mass
        self.age[self.npar] = 0.0
        self.metal[self.npar] = met
        self.npar += 1

    def evolve(self,dt,parameters):

        mloss = 0.0
        N_SNII = 0
        N_SNIa = 0
        for ipar in range(self.npar):
            mstar_min = self.mass_main_sequence(self.age[ipar]+dt,self.metal[ipar])
            mstar_max = self.mass_main_sequence(self.age[ipar],self.metal[ipar])
            mstar_mean = 0.5*(mstar_max+mstar_min)

            # SNII
            if self.with_SNII:
                if (mstar_mean > parameters.SNIImmin) and (mstar_mean < parameters.SNIImmax):
                    n_snii=0
                    num=self.SNII(mstar_min,mstar_max)*self.mass[ipar]
                    if((num-int(num))>random.uniform(0,1)):
                        num += 1
                    if int(num)>=1.0:
                        n_snii = int(num)

                    if n_snii>0:
                        N_SNII += n_snii
                        mloss += n_snii*(0.5*mstar_mean**1.056)

            # OB stars
            if self.with_OB:
                if (mstar_mean > parameters.OBmmin) and (mstar_mean < parameters.OBmmax):
                    mloss += self.OB_wind_mloss(self.age[ipar],self.age[ipar]+dt,self.metal[ipar])

            # AGB
            if self.with_AGB:
                if (mstar_mean > parameters.AGBmmin) and (mstar_mean < parameters.AGBmmax):
                    mloss += self.AGB(mstar_min,mstar_max)*self.mass[ipar]

            # SNIa
            if self.with_SNIa:
                if self.age[ipar] > parameters.SNIatmin:
                    n_snia=0
                    num = self.SNIa(self.age[ipar],self.age[ipar]+dt)*self.mass[ipar]
                    if((num-int(num))>random.uniform(0,1)):
                        num += 1
                    if int(num)>=1.0:
                        n_snia = int(num)

                    if n_snia>0:
                        N_SNIa += n_snia
                        mloss += n_snia*1.4

            self.age[ipar] += dt
            self.mass[ipar] -= mloss
        self.time += dt
        return mloss, N_SNIa, N_SNII

    def mass_main_sequence(self,age,met):
        if age == 0.0:
            return np.inf

        age *= 1e6 # yr
        met = max(7e-5,min(met,3e-2))

        a0=10.13+0.07547*np.log10(met)-0.008084*(np.log10(met))**2
        a1=-4.424-0.7939*np.log10(met)-0.1187*(np.log10(met))**2
        a2=1.262+0.3385*np.log10(met)+0.05417*(np.log10(met))**2

        a0=(-np.log10(age)+a0)

        if(a1**2-4.*a2*a0>=0.0):
            mass=10.0**((-a1-np.sqrt(a1**2-4.0*a2*a0))/(2.0*a2))
        else:
            mass=120.0
        return mass

    def time_main_sequence(self,mass,met):
        """
        Main sequence life time of stars given mass and metallicity.
        Following Raiteri et al. (1996), i.e. Agertz et al. (2013) implementation.
        """
        met = max(7e-5,min(met,3e-2))
        a0 = 10.130 + 0.07547*np.log10(met) - 0.008084*np.log10(met)**2
        a1 = -4.424 - 0.79390*np.log10(met) - 0.118700*np.log10(met)**2
        a2 = 1.2620 + 0.33850*np.log10(met) + 0.054170*np.log10(met)**2

        logt = a0 + a1*np.log10(mass) + a2*np.log10(mass)**2
        return 10**logt/1e6

    def SNII(self,m1,m2):
        A=0.2244557
        return (-A/1.3)*(m2**(-1.3)-m1**(-1.3))

    def AGB(self,m1,m2):
        A=0.2244557
        N1=(m1**(-0.3))*(0.3031/m1-2.97)
        N2=(m2**(-0.3))*(0.3031/m2-2.97)
        return A * (N2-N1)

    def SNIa(self,t1,t2):
        t1 *= 1e6 # yr
        t2 *= 1e6 # yr
        A = 2.6e-13
        return A*(t1/1e9)**(-1.12)*(t2-t1)

    def OB_wind_mloss(self,t1,t2,smet):
        imfboost=0.3143e0/0.224468e0  #K01 to Chabrier

        #--- Fitting parameters ---
        a=0.024357e0*imfboost
        b=0.000460697e0
        ts=1.0e7
        metalscale=a*np.log(smet/b+1.0)
        fmw=0.0

        if(t2<=ts):
            fmw=metalscale*(t2-t1)/ts  #Linear mass loss, m(t2)-m(t1)

        if(t1<ts and t2>ts):
            fmw=metalscale*(ts-t1)/ts  #Linear mass loss, m(t2)-m(t1)

        if(t1>=ts):
            fmw=0.0

        return fmw
