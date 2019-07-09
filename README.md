# Scipy optimization example

## Description

### Problem

It's a curve fitting problem.

### Background

You are given the function `generate_observations`, that generates *n observations* 
sampled from **2 normal distributions**. Each observation generated belongs with probability *P* 
to the first normal distribution. 

### Challenge

Write a function `estimate_parameters` that takes the observations as input and estimates the parameters 
of the normal distributions and probability *P*

### Score

Your score will be the *total variational distance* of your estimated distribution 
and the original distribution. **Get as close as you can to zero.** 