# Intro
Some nifty little python programmes to automate some long
computations.

# Ceff Solver

### Summary
This little solver will take the incidence 
matrix of a directed, grounded graph and either 
calculate the unknown Ceff and node potentials or
the rank of the incidence matrix and give a basis
for the left and right nullspace.

### Dependencies
Only module required is sympy:
```
pip install sympy
```
### Warning
This solver relies on the following:
1. Firstly + node is labelled as node 1 and has potential 1.
2. Secondly - Node is labelled as node 2 and has potential 0.
3. Only one - node and + node exist.
Note: If multiple + and - nodes exist, they can be combined
into a single + and - node since they have the same potential.
4. Also unit conductance in all edges is assumed.


# Particular Integral Solver

### Summary
This solver takes the coefficents of a linear differential
equation at most order 3, with constant coefficients
and then you give it a particular integral and it plugs
it in and gives you the output so you can compare it with RHS.

### Note
Coefficients available for use:
a, b, c, d, e, f

Variables used are y and x.

### Example input
```
a*(x**2)*exp(-6*x)
```

### Dependencies
Again only module required is sympy:
```
pip install sympy
```

# Using GitHub
For anyone that hasn't used github before, you can download
this code using the following steps:
https://www.instructables.com/Downloading-Code-From-GitHub/
