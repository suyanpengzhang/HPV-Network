#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:00:43 2024

@author: suyanpengzhang
"""
import numpy as np
import pandas as pd
# Given Q matrix values
q12, q13, q14, q15 = 2.813e-04, 8.882e-04, 1.305e-04, 1.679e-04
q23, q24, q25 = 9.233e-04, 3.605e-04, 9.211e-08
q34, q35 = 9.364e-04, 2.886e-04
q45 = 1.050e-03

# Time period
t = 365

# Compute transition probabilities using the correct formulas
p11 = np.exp(-(q12 + q13 + q14 + q15) * t)
p12 = (q12 / (q12 + q13 + q14 + q15 - q23 - q24 - q25)) * (np.exp(-(q23+q24+q25) * t) - np.exp(-(q12 + q13 + q14 + q15) * t))

j = q13*q34-q12*q23+q13*(-(q23+q24+q25))
o = -q34**2-(-q12-q13-q14-q15)*q34-(-q12-q13-q14)*(-q23-q24-q25)-(-q23-q24-q25)*q34
x = 1/o
m = (-q12-q13-q14-q15)*q23+q23*q34
l = q12+q13+q14+q15-q23-q24-q25
v = -m/(l*o)
r = (q12*m)/(l*o)-j/o
p13 = j*x*np.exp(-q34-q35*t)+(q12)*v*np.exp((-q23-q24-q25)*t)+r*np.exp((-q12-q13-q14-q15)*t)
#p13 = (q13 / (q23 + q24 + q25-q34-q35)) * (np.exp(-(q34+q35) * t) - np.exp(-(q23 + q24 + q25) * t))
#p14 = (q14 / (q12 + q13 + q14 + q15)) * (1 - np.exp(-(q12 + q13 + q14 + q15) * t))
#p15 = (q15 / (q12 + q13 + q14 + q15)) * (1 - np.exp(-(q12 + q13 + q14 + q15) * t))

p22 = np.exp(-(q23 + q24 + q25) * t)
p23 = (q23 / ((q23 + q24 + q25)-(q34+q35))) * (np.exp(-(q34+q35) * t)-np.exp(-(q23 + q24 + q25) * t))

j = q24*q45-q23*q34+q24*(-(q34+q35))
o = -q45**2-(-q23-q24-q25)*q45-(-q23-q24-q25)*(-q34-q35)-(-q34-q35)*q45
x = 1/o
m = (-q23-q24-q25)*q34+q34*q45
l = q23+q24+q25-q34-q35
v = -m/(l*o)
r = (q23*m)/(l*o)-j/o
p24 = j*x*np.exp(-q45*t)+(q23)*v*np.exp((-q34-q35)*t)+r*np.exp((-q23-q24-q25)*t)



#p25 = (q25 / (q23 + q24 + q25)) * (1 - np.exp(-(q23 + q24 + q25) * t))

p33 = np.exp(-(q34 + q35) * t)
p34 = (q34 / (q34 + q35-q45)) * (np.exp(-(q45) * t) - np.exp(-(q34 + q35) * t))
p35 = 1-np.exp(-(q34 + q35) * t)-(q34 / (q34 + q35-q45)) * (np.exp(-(q45) * t) - np.exp(-(q34 + q35) * t))

p44 = np.exp(-q45 * t)
p45 = 1 - np.exp(-q45 * t)



