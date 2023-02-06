---
author:
- Mitchell Gilmore
date: 2023-02-05
title: EC716 Homework 1
---

# Problem 1.1

![image](plots/Problem_1.svg)

# Problem 1.2

In order to change the sampling rate of signal I would first need to
upsample to the signal by 3. Interpolate the values using a FIR filter
then apply a low pass filter prior to downsampling by a factor of 2.
