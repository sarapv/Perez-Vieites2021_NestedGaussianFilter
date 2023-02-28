# Perez-Vieites2021 - Nested Gaussian Filters (NGFs)

In this repository is included MATLAB code of the NGFs introduced in [[1]](#references). Other algorithms, which have been compared to NHFs in this paper, are also included:

* /NGF (UKF-EKF) : an implementation of the proposed NGF using an unscented Kalman filter (UKF) in the first layer of the filter (i.e., parameter estimation layer) and a bank of extended Kalman filters (EKFs) in the second layer (i.e., state estimation layer).
* /NHF (SMC-EKF) : an implementations of a nested hybrid filter (NHF) [[2]](#references) using sequential Monte Carlo (SMC) in the first layer, and a bank of extended Kalman filters (EKFs) in the second layer.
* /UKF : implementation of an unscented Kalman filter (UKF) [[3]](#references) using state-augmentation techniques (i.e., extending the state vector or variables with the parameters, and adding some artificial dynamics to them).
* /EnKF : implementation of an ensemble Kalman filter (EnKF) [[4]](#references) using state-augmentation techniques.


# References
[1] Pérez-Vieites, S., & Míguez, J. (2021). Nested Gaussian filters for recursive Bayesian inference and nonlinear tracking in state space models. Signal Processing, 189, 108295.

[2] Pérez-Vieites, S., Mariño, I. P., & Míguez, J. (2018). Probabilistic scheme for joint parameter estimation and state prediction in cojmplex dynamical systems. Physical Review E, 98(6), 063305.

[3] Julier, S. J., Uhlmann, J. (2004). Unscented filtering and nonlinear estimation. Proceedings of the IEEE, 92 (2), 401–422.

[4] Evensen, G. (2003). The ensemble Kalman filter: Theoretical formulation and practical implementation. Ocean dynamics, 53 (4), 343–367.
