# Lil script to solve for Ceff and unknown interior node potentials
# given a grounded incidence matrix.
# Note: Assumes only 1 node is grounded

from sympy import *
# Print Matrices Nicely
init_printing()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-------------------- SYMBOL DEFINITIONS ------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Defining our Sympy symbols
Ceff, x3, x4, x5, x6, x7, x8, x9 = symbols('Ceff x3 x4 x5 x6 x7 x8 x9')
node_potentials = [1, 0, x3, x4, x5, x6, x7, x8, x9]

# Instructions
instructions='''
Summary: This little solver will take the incidence 
matrix of a directed, grounded graph and either 
calculate the unknown Ceff and node potentials or
the rank of the incidence matrix and give a basis
for the left and right nullspace.

Warning: This solver relies on the following:
+ Node is labelled as node 1 and has potential 1.
- Node is labelled as node 2 and has potential 0.
Only one - node and + node exist.
Note: If multiple + and - nodes exist, they can be combined
into a single + and - node since they have the same potential.
Also unit conductance in all edges is assumed.
'''
print(instructions)

# Input loop
while True: 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------- MODE --------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("What would you like to solve?")
    print("[1] Ceff and unknown node potentials.")
    print("[2] Calculate rank(A) and left and right nullspace basis.")
    print("")
    try:
        mode_choice = int(input("Mode: "))
        if mode_choice != 1 and mode_choice != 2:
            print("Please input 1 or 2!")
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------- CEFF SOLVER ------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        elif mode_choice == 1:

            # Getting Matrix Input
            print("Incidence Matrix A:")
            R = int(input("No. Rows: ")) 
            C = int(input("No. Cols: "))

            # Initialize our matrix
            A = Matrix([])
            for i in range(R):
                ith_row = []
                for j in range(C):
                    ith_row.append(int(input(f"({i+1}, {j+1}) entry:")))
                A = A.row_insert(i, Matrix([ith_row]))

            # Laplacian
            K = (A.T) * A
            print("")
            print("Laplacian: ")
            pprint(K)
            # print(K)
            # K grounded
            Kg = K
            Kg.col_del(1)
            Kg.row_del(1)
            # Number of rows of K grounded
            n_row = Kg.shape[0] 
            # Number of cols of K grounded
            n_col = Kg.shape[1]
            # Initialize our vector of node potentials
            x = Matrix([[1]])
            for i in range(2, n_col+1):
                x = x.row_insert(i-1, Matrix([[node_potentials[i]]]))

            # Initialize our vector of flux divergence
            # This step is not really necessary
            f = Matrix([[Ceff]])
            for i in range(1, n_col):
                f = f.row_insert(i, Matrix([[0]]))

            # Solutions with respect to ceff
            ceff_solutions = ((Kg**-1) * f) - x
            system = [ceff_solutions.row(i)[:][0] for i in range(0, n_row)]
            variables = [node_potentials[i] for i in range(2, n_col+1)]
            variables.append(Ceff)
            final_solutions = nonlinsolve(system, variables)
            for i in range(len(variables)):
                print("{} = {}".format(variables[i], list(final_solutions)[0][i]))

            quit_input = input("Quit: y/n ")
            if quit_input == 'y':
                print("Bye!")
                break

            elif quit_input == 'n':
                pass

            else:
                print("")
                print("I'll take that as a yes...")
                print("")
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------- NULLSPACE SOLVER --------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        elif mode_choice == 2:

            # Getting Matrix Input
            print("Incidence Matrix:")
            R = int(input("No. Rows: ")) 
            C = int(input("No. Cols: "))
            # Initialize our matrix
            A = Matrix([])
            for i in range(R):
                ith_row = []
                for j in range(C):
                    ith_row.append(int(input(f"({i+1}, {j+1}) entry: ")))
                A = A.row_insert(i, Matrix([ith_row]))

            print("Basis for Right Nullspace: ")
            pprint(A.nullspace())
            A_transpose = A.T
            print("Basis for Left Nullspace: ")
            pprint(A_transpose.nullspace())

            # Calculate rank using Rank nullity theorem
            print("rank(A) = {}".format(C - len(A.nullspace()))) 

            quit_input = input("Quit: y/n ")
            if quit_input == 'y':
                print("Bye!")
                break

            elif quit_input == 'n':
                pass

            else:
                print("")
                print("I'll take that as a yes...")
                print("")
                
        else:
            pass

        # If the input is not valid
    except ValueError:
        print("Please input 1 or 2!")




# Test Incidence Matrix
# A = Matrix([[-1, 0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 0, -1, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, -1, 1, 0],
#             [0, 0, 0, 0, 1, 0, 0, -1, 0],
#             [0, 0, 0, -1, 1, 0, 0, 0, 0],
#             [0, 0, -1, 0, 1, 0, 0, 0, 0],
#             [-1, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 1, -1, 0, 0, 0, 0, 0, 0],
#             [0, -1, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, -1, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, -1, 0, 0, 1],
#             [0, 0, 0, 0, 0, 0, 0, -1, 1]])
