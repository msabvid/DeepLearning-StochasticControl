# Stochastic Control solved using Deep Learning
We solve the control problem, by minimising J
![](/images_readme/control_problem.png)
where g is convex, and by parametrising the policy with a neural network, and using Pontryagin Maximum principle. 
Algorithm:
1. Start with initial policy
2. Solve BSDE using Deep Learning for processes (Y<sub>t</sub>, Z<sub>t</sub>).
3. Update policy by maximising Hamiltonian
4. Go back to 2.

<p align="center">
<img align="middle" src="./numerical_results/trajectories.gif" alt="LQR" width="300" height="250" />
</p>
