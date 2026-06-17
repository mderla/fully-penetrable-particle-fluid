# fully-penetrable-particle-fluid

Metropolis-Hastings sampling of the the canonical ensemble of a simple fluid interacting with a single step potential. Particles are fully penetrable in the sense that it takes finite energy to get any given particle pair to zero distance.

## What is happening here?

In the code we perform a metropolis-hastings sampling of the canonical ensemble for a system of *penetrable spheres* (PS) and *penetrable disks* (PD). The "penetrable" is meant in the sense that they interact via the pair potential $u_E(r)=E\,\Theta(\sigma-r)$. This treatment of penetrable spheres/disks can be understood as a variation on the problem of predicting the free energy of a system of *rigid spheres* (RS) or *rigid disks* (RD), also referrred to as hard spheres or hard disks: when $E\gg k_\text{B}T$, a pair of particles would need an unrealistically high energy to overcome the barrier $E$ to closing in to any distance shorter than $\sigma$, effectively making them rigid. The family $\{u_E\}_{E\in[0,\infty)}$ in this sense connects the ideal gas, interacting with $u_0(r)\equiv 0$, to the rigid particle fluid, doing so with potentials that have a clear mathematical resemblance with rigid particle interaction.

The *overlap energy* $E$ (or *penetration energy barrier*) is the only energy scale in the system besides temperature, such that the phase diagram only depends on the ratio $$\varepsilon=\frac{E}{k_\text{B}T}$$ Besides the reduced overlap energy $\varepsilon$, the system behaviour is still shaped by the *packing density* $$\Phi=B_d\left(\frac{\sigma}{2}\right)^d\rho$$ where $\rho$ is number density and $B_d$ the volume of a unit sphere in $d$ dimensions (which reads, forexample, $B_2=\pi$ and $B_3=\frac{4}{3}\pi$ in $d=3$).

## For now ...

Code development is on low flame. As soon as it is reliable I seek to find and explain the various behaviour regimes observed in simulation.

# Copyright notice

&copy; 2026 Miriam Derla - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)