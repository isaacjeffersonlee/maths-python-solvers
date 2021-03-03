# Matrix Solver                                                             
### Author: Isaac Lee

### Summary
Advanced matrix (and other things) solver, with the following features:

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
10          | Delete rows/cols
11          | Applied Maths Mode: Ceff Solver, Method of Relaxation, Spring Mass Systems
12          | Calculus Mode: Systems of Differential Equations, ODE Solver
13          | Save/Load matrices from file

### Advantages
Similar tools exist online such as symbolab or wolfram alpha,
however my solver has a big advantage in that it saves every matrix output
after each operation, and so multiple operations can be easily performed
on the same matrix one after another. Also I have implemented specialist
modes for Applied maths and Calculus which make large matrix input simple.

### Inputs:

Input can be integer, decimal, fractional, symbolic inputs and functional input.
By functional input I mean cos(x), sin(x), exp(-4*x**2), e.t.c. 
(Which get rendered using unicode). Because of how floats work in python
I would advice against using any decimal input, instead use a fraction.
Also can take greek letters, i.e typing in alpha will render as: α, or 
phi_1 will render as: ϕ₁
                
### Errors:

Sympy interprets certain Greek letters such as beta and gamma as functions
which will cause errors so if using greek letters and you get:
"TypeError: unsupported operand type(s) for *: 'FunctionClass' and 'Symbol'"
Then don't use those letters.

The jupyter notebook version occasionally closes the 
input field randomly. If this happens just stop
the cell and re-run it.

### Contents of this repository:
Note: when downloading/running this repository, make sure to have all files/directories
in the same directory because they depend on each other.

Images/ -> Contains phase portrait images.

solver_functions.py -> main code function definitions, don't run this file directly.

matrix_solver.py -> base version, can be run as you would a normal python .py file.

matrix_solver_jupyter.ipynb -> jupyter version, best version imo, 
can print output in nicely formatted LaTeX and display images.

memory.txt -> stores saved matrices for persistant use.

### Available Modes:

##### [1].  Multiply Matrices
Either multiply A*B or A^n for any input n.
Input matrices saved.

##### [2].  Inverse
Find the inverse of A and save it.

##### [3].  Transpose
Find the transpose of A and save it.

##### [4].  Determinant
Find the determinant of A.

##### [5].  Row reduce to echelon form
Row reduce A to row reduced echelon form and 
save the result.

##### [6].  Basis for left and right nullspace
Find a basis for the nullspace of A and A transpose.

##### [7].  Rank and Basis for Row space and Column space
Find the rank of A and the basis for it's row 
and columns spaces.

##### [8].  Eigenvalues and Eigenvectors
Find the eigenvalues and eigen vectors of A.
Note: Solutions are presented in triplets:
[eigen value, algebraic multiplicity, eigen vector],
where the algebraic multiplicity is the number of times
the eigen value is a solution for the characteristic 
equation, i.e a repeat root would have algebraic
multiplicity = 2.

##### [9].  Diagonalize/Jordan Normal
Either A is diagonalizable and there exists matrices
P and D such that PDP^-1 = A, so calculate these and print
them, or A is not diagonalizable, in which case finds
the jordan normal form of A and P s.t PJP^-1 = A.

##### [10]. Delete rows/cols
Delete rows and columns of a new matrix or a saved matrix.
Note: This operation will permanently change any saved
matrices it operates on.

##### [11]. Applied Maths Mode
Contains several sub-modes:

###### [1]. Ceff Solver
Takes an incidence matrix for a graph and a vector 
of node potentials and vector of net flux divergence
and finds the unknown node potentials and Ceff. 

###### [2]. Method of relaxation iterator
Takes in space seperated connected node numbers,
then generates an adjacency matrix, then uses the 
method of relaxation to iterate a given number of times,
printing a list of updated node potentials in both 
fractional and decimal form after each iteration.

###### [3]. Spring System Solver
Equilibrium:
Again takes a matrix and vectors of external forces
and unknown displacements and solves for the unknown
displacements in terms of the external forces,
then solves for the unknown reaction forces at the walls.

Non-equilibrium:
Takes incidence matrix, unknown displacements and external forces
and solves for the displacements as a function of time using
Newtons Second Law. Also gives eigenvectors and eigenvalues.

##### [12]. Calculus Mode: Differential Equations
###### [1]. System ODEs Solver
Solves systems of the form dY/dt = A*Y + g(t)
where Y is the vector of functions of t, i.e [x(t), y(t)],
A is a square matrix and g(t) is the non-homogeneous part vector.
Finds, eigen values, eigen vectors, P, D/J s.t PDP^-1 = A,
i.e diagonalizes/jordanizes A, then solves for the general 
solutions of x(t), y(t), ... Also for 2x2 matrices gives 
the determinant trace quadrant for use in plotting the phase 
portrait of the system, and then displays an image from the 
notes of what general shape the phase portrait will take.
Warning: This image is not a plot of the exact system,
rather just a screenshot from the notes demonstrating what 
a phase portrait in that quadrant should look like.

###### [2]. Singular ODE Solver
Takes in the dependent and independent variables, i.e y and x,
the order of the ODE, the coefficients and the non-homogeneous RHS,
(Right Hand Side) and returns the solutions to both the homogeneous
and non-homogeneous parts.

##### [13]. Save/Load matrices from file
This solver uses a memory.txt file located in the same 
directory to the list of saved matrices from the user.
This mode allows the user to save the current sessions
saved matrices to the file for persistant saving/loading 
of matrices. If there is an error relating to this,
make sure there is a file called memory.txt in the same
directory as this file.


### Jupyter Version
This repo has both a .py and .ipynb version. The jupyter version
has the ability to print in nicely formatted LaTeX and show images
so is probably my preferred version, although it is also slightly slower.

### Errors:
If an error occurs, or something takes too long to run,
just interupt the process/stop the kernel and start again.

### Dependencies
The following are libraries used:
- sympy 
- fractions 
- pickle
- os
- sys 
- IPython.display 
- jupyter 
However I'm pretty sure the only libraries
not in the base python install are sympy and jupyter:
```
pip install sympy

pip install jupyter
```

# Using GitHub
For anyone that hasn't used github before, you can download
this code using the following steps:
https://www.instructables.com/Downloading-Code-From-GitHub/
