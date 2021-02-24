# Intro
Some nifty little python programs to automate some long
computations.

# Matrix Solver                                                             

### Summary
Advanced matrix calculator, with the following features:

Available Operations:
Mode No.    | Mode
------------| -------------
1           | Multiply Matrices
2           | Inverse
3           | Transpose
4           | Determinant
5           | Row reduce to echelon form.
6           | Basis for left and right nullspace.
7           | Rank and Basis for Row space and Column space
8           | Eigenvalues and Eigenvectors
9           | Diagonalize/Jordan Normal

### Advantages
Similar tools exist online such as symbolab or wolfram alpha,
however my solver has a big advantage in that it saves every matrix output
after each operation, and so multiple operations can be easily performed
on the same matrix one after another.

### Inputs:
Matrix can take integer, fractional and symbolic inputs. 
E.g 1, 2, 3, 1/2, 11/12, a, b, c, pi, alpha, beta, gamma, delta, e.t.c. 
(Which get rendered using unicode/LaTeX depending on version). 

Because of how floats work in python I have opted to not allow decimal input,
any decimal input will be truncated to it's integer part. If you want a decimal input,
i.e 1.2, use a fractional input instead, i.e 6/5.

### Jupyter Version
This repo has both a .py and .ipynb version. The jupyter version
has the ability to print in nicely formatted LaTeX so is probably my
preferred version, although it is also slightly slower.

### Errors:
If an error occurs, or something takes too long to run,
just interupt the process and start again.


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
