# Perez-Vieites2021 - Nested Gaussian Filters (NGFs)

In this repository is included MATLAB code of the NGFs introduced in [[1]](#references). Other algorithms, which have been compared to NHFs in this paper, are also included:

* /NGF (UKF-EKF) : an implementation of the proposed NGF using an unscented Kalman filter (UKF) in the first layer of the filter (i.e., parameter estimation layer) and a bank of extended Kalman filters (EKFs) in the second layer (i.e., state estimation layer).
* /NHF (SMC-EKF) : an implementations of a nested hybrid filter (NHF) [[2]](#references) using sequential Monte Carlo (SMC) in the first layer, and a bank of extended Kalman filters (EKFs) in the second layer.
* /UKF : implementation of an unscented Kalman filter (UKF) [[3]](#references) using state-augmentation techniques (i.e., extending the state vector or variables with the parameters, and adding some artificial dynamics to them).
* /EnKF : implementation of an ensemble Kalman filter (EnKF) [[4]](#references) using state-augmentation techniques.

The running file is **main_script.m**. It is necessary to add all the folders into the current MATLAB path in order to call the fuctions in them.

# References
[1] Pérez-Vieites, S., & Míguez, J. (2021). Nested Gaussian filters for recursive Bayesian inference and nonlinear tracking in state space models. Signal Processing, 189, 108295.

[2] Pérez-Vieites, S., Mariño, I. P., & Míguez, J. (2018). Probabilistic scheme for joint parameter estimation and state prediction in cojmplex dynamical systems. Physical Review E, 98(6), 063305.

[3] Julier, S. J., Uhlmann, J. (2004). Unscented filtering and nonlinear estimation. Proceedings of the IEEE, 92 (2), 401–422.

[4] Evensen, G. (2003). The ensemble Kalman filter: Theoretical formulation and practical implementation. Ocean dynamics, 53 (4), 343–367.



# MIT License

Copyright (c) 2023 Sara Pérez Vieites

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
