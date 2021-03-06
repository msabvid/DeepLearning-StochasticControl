# Stochastic Control solved using Deep Learning
We solve the control problem, by minimising J
![](/images_readme/control_problem.png)
where g is convex. The policy alpha is parametrised with a neural network, and we use Method of successive approximations on Pontryagin Maximum principle. 
Algorithm:
1. Start with initial policy
2. Solve BSDE using Deep Learning for processes (Y<sub>t</sub>, Z<sub>t</sub>).
3. Update policy by maximising Hamiltonian (analog to Q-learning on model-free RL)
4. Go back to 2.

<p align="center">
<img align="middle" src="./numerical_results/trajectories.gif" alt="LQR" width="300" height="250" />
</p>


## TODO

Code is loopy. The bsde solver and the Hamiltonian should be vectorized across time. 
