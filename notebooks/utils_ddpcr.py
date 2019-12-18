import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import csv
import os
from scipy.stats import norm


def poisson_correct_counts(Np,Nn):
    #Calculates real number of molecules from Np positive wells and Nn negative wells
    Nt=Nn+Np
    return -math.log((Nt-Np)/Nt)*Nt

def poisson_correct_prob(Np,Nn):
    #Calculates probability of finding a molecule in a well from Np positive wells and Nn negative wells
    Nt=Nn+Np
    return -math.log((Nt-Np)/Nt)

def poisson_counts_upper_bound(Np,Nn,z=1):
    #Calculates upper for real number of Np positive wells and negative wells.
    #Default is 1 SD, parameter z defines how many SD away
    Nt=Nn+Np
    p=Np/Nt
    N_upper = Nt* ( -math.log(1-(p+z*math.sqrt(p*(1-p)/Nt))) )
    return N_upper

def poisson_counts_lower_bound(Np,Nn,z=1):
    #Calculates lower bound for real number of Np positive wells and negative wells.
    #Default is 1 SD, parameter z defines how many SD away
    Nt=Nn+Np
    p=Np/Nt
    N_lower = Nt* ( -math.log(1-(p-z*math.sqrt(p*(1-p)/Nt))) )
    return N_lower

def poisson_sd_from_bounds(Np, Nn):
    #Calculates 'average' SD from lower and higher bounds from Poisson statistics
    #done using z=1
    z=1
    return (poisson_counts_upper_bound(Np,Nn,z) - poisson_counts_lower_bound(Np,Nn,z)) / 2.

def allele_counts(df, sample_list, probe_name):
    SNP_min_droplets=500
    filterwells = ((df.Target == probe_name) &
               (df.Sample.isin(sample_list))&
               (df['Ch1+Ch2+']+df['Ch1+Ch2-']+df['Ch1-Ch2+']+df['Ch1-Ch2-'] > SNP_min_droplets))
    allele_df = df.loc[filterwells, :]
    return poisson_correct_counts(allele_df.sum()['Positives'],allele_df.sum()['Negatives'])

def gauss_joan(x,mu,sd):
    return 1/math.sqrt(2*sd**2*math.pi)*math.exp(-(((x-mu)/sd)**2)/2)

def wei_zvalues_xlinked(Nm,Nwt,FF):
    Nt= Nm + Nwt
    half_fetal = 0.5*FF/100
    z_positive = Nt*FF/100/(1-half_fetal)/math.sqrt(Nt)
    z_negative = 0
    z_experiment = (Nm-Nwt+Nt*half_fetal/(1-half_fetal))/math.sqrt(Nt)
    return z_positive, z_negative, z_experiment

def allele_fractions_xlinked(Nm,Nwt,FF,Error_FF):
    res={}
    Nt= Nm + Nwt
    half_fetal = 0.5*FF/100
    res['Nt']=Nt
    res['frac_positive'] = 0.5/(1-half_fetal)
    res['frac_negative'] = 0.5*(1.-FF/100)/(1.-0.5*FF/100)
    res['SD_positive'] = math.sqrt((Nt)*res['frac_positive'])/Nt
    res['SD_negative'] = math.sqrt((Nt)*res['frac_negative'])/Nt
    res['frac_experiment'] = Nm/(Nm+Nwt)
    res['SD_FF'] = Error_FF * 1/(2-FF)**2
    res['TotErr_pos'] = math.sqrt((res['SD_FF'])**2+(res['SD_positive'])**2)
    res['TotErr_neg'] = math.sqrt((res['SD_FF'])**2+(res['SD_negative'])**2)
    res['up_bound'], res['low_bound'] = SPRT_limits(res['frac_negative'],res['frac_positive'],Nt)
    return res



def print_xlinked(Nm,Nwt,FF,Error_FF):
    z_positive, z_negative, z_experiment = wei_zvalues_xlinked(Nm,Nwt,FF)
    res = allele_fractions_xlinked(Nm,Nwt,FF,Error_FF)
    print('Mutant counts {:.1f}, WT counts {:.1f}, Total counts {:.1f}'.format(Nm,Nwt,res['Nt']))
    print('z-value {:.2f} \n\tPositive expected: {:.2f} \n\tNegative expected: {:.2f}'
          .format(z_experiment,z_positive, z_negative))
    print('Mutant Allele Frac {:.3f} \n\tPositive Frac Expected: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})\
    \n\tNegative Frac Expected:: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})'
          .format(res['frac_experiment'], res['frac_positive'], res['SD_positive'],\
                  res['TotErr_pos'], res['frac_negative'], res['SD_negative'], res['TotErr_neg']))
    print('\tSPRT low bound {:.3f}\n \tSPRT high bound: {:.3f}.'.format(res['low_bound'],res['up_bound']))
    if res['frac_experiment']>res['up_bound']:
        print('SPRT classifier: Fetus is affected.')
    elif res['frac_experiment']<res['low_bound']:
        print('SPRT classifier: Fetus is non-affected.')
    else:
        print('SPRT classifier: Indeterminate result.')


def wei_zvalues_autosomalrec(Nm,Nwt,FF):
    Nt= Nm + Nwt
    z_positive = Nt*FF/100/math.sqrt(Nt)
    z_negative = 0
    z_experiment = (Nm-Nwt)/math.sqrt(Nt)
    return z_positive, z_negative, z_experiment

def allele_fractions_autosomalrec(Nm,Nwt,FF,Error_FF):
    res={}
    Nt= Nm + Nwt
    res['Nt']=Nt
    res['frac_positive'] = 0.5*(1.+FF/100)
    res['frac_negative'] = 0.5
    res['SD_positive'] = math.sqrt((Nt)*res['frac_positive'])/Nt
    res['SD_negative'] = math.sqrt((Nt)*res['frac_negative'])/Nt
    res['frac_experiment'] = Nm/(Nm+Nwt)
    res['SD_FF'] = Error_FF/100 * 0.5
    res['TotErr_pos'] = math.sqrt((res['SD_FF'])**2+(res['SD_positive'])**2)
    res['TotErr_neg'] = res['SD_negative'] # the fetal fraction plays no role for heterozyogous baby
    res['up_bound'], res['low_bound'] = SPRT_limits(res['frac_negative'],res['frac_positive'],Nt)
    return res

def print_autosomalrec(Nm,Nwt,FF,Error_FF):
    z_positive, z_negative, z_experiment = wei_zvalues_autosomalrec(Nm,Nwt,FF)
    res = allele_fractions_autosomalrec(Nm,Nwt,FF,Error_FF)
    print('Mutant counts {:.1f}, WT counts {:.1f}, Total counts {:.1f}'.format(Nm,Nwt,res['Nt']))
    print('z-value {:.2f} \n\tPositive expected: {:.2f} \n\tNegative expected: {:.2f}'
          .format(z_experiment,z_positive, z_negative))
    print('Mutant Allele Frac {:.3f} \n\tPositive Frac Expected: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})\
    \n\tNegative Frac Expected:: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})'
          .format(res['frac_experiment'], res['frac_positive'], res['SD_positive'],\
                  res['TotErr_pos'], res['frac_negative'], res['SD_negative'], res['TotErr_neg']))
    print('\tSPRT low bound {:.3f}\n \tSPRT high bound: {:.3f}.'.format(res['low_bound'],res['up_bound']))
    if res['frac_experiment']>res['up_bound']:
        print('SPRT classifier: Fetus is affected.')
    elif res['frac_experiment']<res['low_bound']:
        print('SPRT classifier: Fetus is non-affected.')
    else:
        print('SPRT classifier: Indeterminate result.')


def SPRT_limits(q0,q1,Nt):
    #calculates SPRT classifier limiers (D LO) from expected fractions and total number of counts
    d=(1-q1)/(1-q0)
    g=q1/q0*(1-q0)/(1-q1)
    up_bound=(math.log(8)/Nt-math.log(d))/math.log(g)
    low_bound=(math.log(1/8)/Nt-math.log(d))/math.log(g)
    return up_bound,low_bound



#HERE THE INCORRECT WAYS TO DO IT, when we forget that the fetal fraction can not be plugged directly with a reduced
#number of counts as we're only measuring NT*(1-eps/2), we are actually measuring real NT=[NT*(1+eps/2)]/(1-eps/2)
def allele_fractions_xlinked_bad(Nm,Nwt,FF):
    Nt= Nm + Nwt
    half_fetal = 0.5*FF/100
    frac_positive = 0.5+0.5*half_fetal
    frac_negative = 0.5-0.5*half_fetal
    SD_positive = math.sqrt((Nt)*frac_positive)/Nt
    SD_negative = math.sqrt((Nt)*frac_negative)/Nt
    frac_experiment = Nm/(Nm+Nwt)
    return frac_experiment, frac_positive, frac_negative, SD_positive, SD_negative,


def wei_zvalues_xlinked_bad(Nm,Nwt,FF):
    Nt= Nm + Nwt
    half_fetal = 0.5*FF/100
    z_positive = Nt*FF/100/math.sqrt(Nt)
    z_negative = 0
    z_experiment = (Nm-Nwt+Nt*half_fetal)/math.sqrt(Nt)
    return z_positive, z_negative, z_experiment

def allele_fractions_xlinked_DLo(Nm,Nwt,FF):
    Nt= Nm + Nwt
    half_fetal = 0.5*FF/100
    frac_positive = 1/(2-FF/100)
    frac_negative = (1-FF/100)/(2-FF/100)
    SD_positive = math.sqrt((Nt)*frac_positive)/Nt
    SD_negative = math.sqrt((Nt)*frac_negative)/Nt
    frac_experiment = Nm/(Nm+Nwt)
    return frac_experiment, frac_positive, frac_negative, SD_positive, SD_negative

### Functions for carrier maternal

def wei_zvalues_carrier_maternal(Nm,Nwt,FF):
    Nt= Nm + Nwt
    z_positive = Nt*FF/100/math.sqrt(Nt)
    z_negative = 0
    z_experiment = (Nm-Nwt)/math.sqrt(Nt) + Nt*FF/100/math.sqrt(Nt)
    return z_positive, z_negative, z_experiment

def allele_fractions_carrier_maternal(Nm,Nwt,FF,Error_FF):
    res={}
    Nt= Nm + Nwt
    res['Nt']=Nt
    res['frac_positive'] = 0.5
    res['frac_negative'] = 0.5*(1.-FF/100)
    res['SD_positive'] = math.sqrt((Nt)*res['frac_positive'])/Nt
    res['SD_negative'] = math.sqrt((Nt)*res['frac_negative'])/Nt
    res['frac_experiment'] = Nm/(Nm+Nwt)
    res['SD_FF'] = Error_FF/100 * 0.5
    res['TotErr_pos'] = math.sqrt((res['SD_FF'])**2+(res['SD_positive'])**2)
    res['TotErr_neg'] = res['SD_negative'] # the fetal fraction plays no role for heterozyogous baby
    res['up_bound'], res['low_bound'] = SPRT_limits(res['frac_negative'],res['frac_positive'],Nt)
    return res

def print_carrier_maternal(Nm,Nwt,FF,Error_FF):
    z_positive, z_negative, z_experiment = wei_zvalues_carrier_maternal(Nm,Nwt,FF)
    res = allele_fractions_carrier_maternal(Nm,Nwt,FF,Error_FF)
    print('Mutant counts {:.1f}, WT counts {:.1f}, Total counts {:.1f}'.format(Nm,Nwt,res['Nt']))
    print('z-value {:.2f} \n\tPositive expected: {:.2f} \n\tNegative expected: {:.2f}'
          .format(z_experiment,z_positive, z_negative))
    print('Mutant Allele Frac {:.3f} \n\tPositive Frac Expected: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})\
    \n\tNegative Frac Expected:: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})'
          .format(res['frac_experiment'], res['frac_positive'], res['SD_positive'],\
                  res['TotErr_pos'], res['frac_negative'], res['SD_negative'], res['TotErr_neg']))
    print('\tSPRT low bound {:.3f}\n \tSPRT high bound: {:.3f}.'.format(res['low_bound'],res['up_bound']))
    if res['frac_experiment']>res['up_bound']:
        print('SPRT classifier: Fetus is carrier of maternal mutation.')
    elif res['frac_experiment']<res['low_bound']:
        print('SPRT classifier: Fetus is non-carrier of maternal mutation.')
    else:
        print('SPRT classifier: Indeterminate result.')


#### functions for carrier paternal

def wei_zvalues_carrier_paternal(Nm,Nwt,FF):
    Nt= Nm + Nwt
    Nexpect = Nt*FF/2/100
    z_positive = math.sqrt(Nexpect)
    z_negative = 0
    z_experiment = math.sqrt(Nm)
    return z_positive, z_negative, z_experiment

def allele_fractions_carrier_paternal(Nm,Nwt,FF,Error_FF):
    #For paternal mutation non-carrier we introduce 1 counts of potential background
    res={}
    background_counts=1.
    Nt= Nm + Nwt
    res['Nt']=Nt
    res['frac_positive'] = 0.5*FF/100
    res['frac_negative'] = background_counts / Nt
    res['SD_positive'] = math.sqrt((Nt)*res['frac_positive'])/Nt
    res['SD_negative'] = math.sqrt((Nt)*res['frac_negative'])/Nt
    res['frac_experiment'] = Nm/(Nm+Nwt)
    res['SD_FF'] = Error_FF/100 * 0.5
    res['TotErr_pos'] = math.sqrt((res['SD_FF'])**2+(res['SD_positive'])**2)
    res['TotErr_neg'] = res['SD_negative'] # the fetal fraction plays no role for heterozyogous baby
    res['up_bound'], res['low_bound'] = SPRT_limits(res['frac_negative'],res['frac_positive'],Nt)
    return res

def print_carrier_paternal(Nm,Nwt,FF,Error_FF):
    z_positive, z_negative, z_experiment = wei_zvalues_carrier_paternal(Nm,Nwt,FF)
    res = allele_fractions_carrier_paternal(Nm,Nwt,FF,Error_FF)
    print('Mutant counts {:.1f}, WT counts {:.1f}, Total counts {:.1f}'.format(Nm,Nwt,res['Nt']))
    print('z-value {:.2f} \n\tPositive expected: {:.2f} \n\tNegative expected: {:.2f}'
          .format(z_experiment,z_positive, z_negative))
    print('Mutant Allele Frac {:.3f} \n\tPositive Frac Expected: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})\
    \n\tNegative Frac Expected:: {:.3f} (SD_N:{:.3f} Err_FF:{:.3f})'
          .format(res['frac_experiment'], res['frac_positive'], res['SD_positive'],\
                  res['TotErr_pos'], res['frac_negative'], res['SD_negative'], res['TotErr_neg']))
    print('\tSPRT low bound {:.3f}\n \tSPRT high bound: {:.3f}.'.format(res['low_bound'],res['up_bound']))
    if res['frac_experiment']>res['up_bound']:
        print('SPRT classifier: Fetus is carrier of paternal mutation.')
    elif res['frac_experiment']<res['low_bound']:
        print('SPRT classifier: Fetus is non-carrier of paternal mutation.')
    else:
        print('SPRT classifier: Indetermintate. Mutation detected but counts much lower than expected from Fetal Fraction.')
