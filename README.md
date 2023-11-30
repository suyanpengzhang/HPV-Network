# HPV Vaccination Attitude Analysis in Kenya

## Description
This repository hosts an innovative project that analyzes household survey data from Kenya, focusing on demographics, openness to HPV vaccination, and prevailing social norms. Utilizing this data, we construct a bidirectional BA network and apply the linear threshold model with two thresholds to capture both negative and positive attitudes towards HPV vaccination.

## Features
1. **Simulation Analysis**: Utilizes `simulation.py` to generate network structures from the data and conducts sensitivity analyses through simulations with varying thresholds.
2. **Network Visualization**: Employs `OpenGLVisual.py` for visualizing network evolution. Currently, this supports partial data sets (30 out of 1000 nodes).
3. **Optimization Problem**: Addresses an optimization problem involving budget constraints and inhomogeneous education costs to shift household attitudes to positive. The relevant scripts are `branchandcut.py` for heuristic branch-and-cut methods and `HPV_optimization.py` for general optimization.

## Technology Stack
- Python
- NetworkX for network analysis
- NumPy and Pandas for data manipulation
- Matplotlib and OpenGL for visualization

## Contact Information
If you have any questions or would like to collaborate on this project, please contact the project maintainer at [suyan430@gmail.com].
