import numpy as np
from matplotlib import pyplot as plt
from sympy import *
from sympy.matrices.common import MatrixError
from fractions import Fraction
import pickle
import os
from sys import platform
from IPython.display import Image, Math, Latex, clear_output


class Solvers:
    def __init__(self, using_jupyter, saved_matrices):
        self.using_jupyter = using_jupyter
        self.saved_matrices = saved_matrices


    def dprint(self, object_to_print):
        """
        When using an ipython notebook we want to use the
        display() function to print matrices, but when 
        we're using the .py version we just want pprint().
        This function checks which version is being used and
        uses the appropriate function to print input string.
        """
        using_jupyter = self.using_jupyter
        if using_jupyter == True:
            display(object_to_print)

        elif using_jupyter == False:
            pprint(object_to_print)

        else:
            print("Error in dprint function!")


    def clear_previous(self):
        """Check which operating system and clear the output accordingly"""
        # Linux or MacOS
        using_jupyter = self.using_jupyter
        if using_jupyter: # If being called from jupyter version
            clear_output(wait=False)
        else:
            if platform == "linux" or platform == "linux32" or platform == "darwin":
                os.system('clear')
            # Windows
            elif platform == "win32":
                os.system('cls')

    def save_matrix(self, matrix_to_save):
        """ Save a matrix. """
        saved_matrices = self.saved_matrices
        if matrix_to_save not in saved_matrices:
            saved_matrices.append(matrix_to_save)
            print("")


    def use_saved_matrices(self):
        """ Ask whether to use a saved matrix. """
        print("")
        use_saved = input("Use a saved matrix? y/n ")
        if use_saved == 'y':
            return True
        else:
            return False


    def print_saved_matrices(self):
        """ 
        Print out available saved matrices along with their indexes.
        Also returns False if no matrices are saved and True if atleast
        1 saved matrix exists.
        """
        saved_matrices = self.saved_matrices
        if not saved_matrices:
            print("No saved matrices!")
            return False
        else:
            saved_with_indexes = [([i], saved_matrices[i]) for i in range(len(saved_matrices))]
            self.dprint(saved_with_indexes)
            return True


    def simplify_matrix(self, A):
        """Simplify a matrix"""
        A_rows = A.shape[0]
        A_cols = A.shape[1]
        B = zeros(A_rows, A_cols) # Initiliaze zero matrix
        for i in range(A_rows):
            for j in range(A_cols):
                B[i,j] = A[i,j].simplify()
        return B


    def get_input(self, prompt_text):
        """
        Takes both numerical and variable input 
        and returns the result as either a float or 
        a sympy symbol.
        Note: Also can take fractional input, e.g '1/2'.
        Note: For decimal input use fractions, if a float is given the
        integer part is used.
        """
        input_element = input(prompt_text)
        return sympify(input_element)


    def input_adjacency(self):
        """
        Input: Connected Node Numbers, 
        Returns: Adjacency Matrix
        """
        while True: # Get node_num loop
            try:
                node_num = int(input("Total number of nodes: "))
                break
            except ValueError:
                print("Please input a positive integer!")
                continue

        while True:
            try:
                A = Matrix([]) # Initialize A
                print("Input connected nodes, seperated by a space.")
                print("Note: It is assumed the first node is labelled 1 and not 0.")
                print("E.g if node 1 is connected to nodes 3, 4, and 5 => input: 3 4 5")
                for i in range(node_num):
                    ith_row_node_idx_string = input("Nodes connected to node {}: ".format(i+1))
                    ith_row_node_idx = [int(node)-1 for node in ith_row_node_idx_string.split(" ")]
                    ith_row_list = [0] * node_num
                    for idx in ith_row_node_idx:
                        ith_row_list[idx] = 1

                    A = A.row_insert(i, Matrix([ith_row_list]))

                return A


            except ValueError:
                print("")
                print("Error: Please input a positive integer!")
                print("")
                continue


    def harmonic_iterator(self, start_potentials, boundary_node_idx, A, n):
        """
        Takes a list of 1s and 0s representing the starting
        potentials for the method of relaxation and performs
        n iterations and returns the updated list of potentials.
        Args: start_potentials, type == list, starting potentials for nodes
        Args: boundary_node_idx, type == list, indexes of the boundary nodes, 
        which don't get touched by the iterations.
        Args: A, type == Sympy Matrix, Adjacency Matrix, consisting of 1s and 0s
        Args: n, type == positive integer
        """
        potentials = start_potentials
        print("Starting node potentials:")
        self.dprint(Matrix([start_potentials]))
        print("")
        for iteration in range(n):
            for i in range(len(potentials)): # loop through each potential
                if i in boundary_node_idx: # don't change boundary node potentials
                    pass
                else: # interior nodes
                    # nodes connected to node i
                    connected_node_idx = [j for j in range(len(A.row(i))) if A[i,j] == 1] 
                    connected_node_sum_potentials = sum([potentials[j] for j in connected_node_idx]) 
                    # Update node i potential with the mean of the potentials of the connected nodes
                    potentials[i] = Fraction(connected_node_sum_potentials, len(connected_node_idx))
                    # print("Node {} potential, iteration {}: {}".format(i+1, iteration+1, potentials[i]))
            print("Updated node potentials after iteration {}:".format(iteration+1))
            pprint(Matrix([potentials])) # pretty print this instead of display, much faster
            print("")
            print([round(float(potential), 4) for potential in potentials])
            print("")

        return potentials


    def input_matrix(self):
        """ 
        Either use a saved matrix or prompt for a new one.
        Returns the  matrix A.
        Returns None if an invalid saved matrix index is given.
        """
        saved_matrices = self.saved_matrices
        while True:
            if self.use_saved_matrices() and self.print_saved_matrices():
                chosen_matrix_number_string = input("Use saved matrix No.: ")
                try:
                    chosen_matrix_number = int(chosen_matrix_number_string)
                    A = saved_matrices[chosen_matrix_number]
                    return A

                except (IndexError, ValueError) as e:
                    print("")
                    print("Error: {} is not a valid saved matrix index".format(chosen_matrix_number_string))
                    print("")
                    pass
                    # Returns None
            else:
                # Getting Matrix Input

                try:
                    # Initialize our matrix.
                    print("")
                    print("New Matrix: ")
                    row_num = int(input("No. Rows: ")) 
                    col_num = int(input("No. Cols: "))
                    print("Input each row as a space seperated list of the elements.")
                    print("E.g: 1 0 0 0 -1 0")
                    A = Matrix([])
                    # Row wise input
                    for i in range(row_num):
                        ith_row = sympify(input("Row {}: ".format(i+1)).split(" "))
                        A = A.row_insert(i, Matrix([ith_row]))
                        # Element wise input
                        # for j in range(C):
                        #     element = self.get_input("({}, {}) entry: ".format(i+1, j+1))
                        #     ith_row.append(element)
                        # A = A.row_insert(i, Matrix([ith_row]))
                    self.save_matrix(A)
                    return A

                except ValueError:
                    print("Not a valid input!")
                    continue
                    # Returns None


    def print_eigen(self, A):
        """Simplify and print eigen values, algebraic multiplicity and eigenvectors of A"""
        eigen_triples = A.eigenvects()
        print("")
        print("(Eigenvalue, [Algebraic Multiplicity], Eigenspace Basis):")
        self.dprint([(simplify(eigen_triple[0]), [eigen_triple[1]], [self.simplify_matrix(eigen_vect) \
                                                                for eigen_vect in eigen_triple[2]]) \
                for eigen_triple in eigen_triples])
        print("")


    def diagonalize_or_jordanize(self, A):
        """
        Try to diagonalize input matrix A, if not use jordan normal form,
        Returns P and D/J s.t PDP**-1 = A
        """
        try: 
            P, D = A.diagonalize()
            print("")
            print("Matrix is diagonalizable!")
            print("P D P**-1 == A:")
            self.dprint([P, D, P**-1, A])
            print("")
            P = self.simplify_matrix(P)
            D = self.simplify_matrix(D)
            return {'P' : P, 'D' : D}

        except NonSquareMatrixError:
            print("")
            print("Error: Not a square matrix => Not diagonalizable!")
            print("")

        except MatrixError:
            print("")
            print("Matrix is not diagonalizable!")
            print("Trying Jordan Normal Form...")
            print("")
            P, J = A.jordan_form()
            print("P J P**-1 = A:")
            P = self.simplify_matrix(P)
            J = self.simplify_matrix(J)
            self.dprint([P, J, P**-1, A])
            print("")
            self.save_matrix(P)
            self.save_matrix(J)
            return {'P' : P, 'D' : J}


    def mode_selector(self):
        """
        Prints different available modes.
        Gets input.
        Checks for invalid input.
        """ 
        print("")
        print("Available Operations:")
        print("[0].  Quit")
        print("[1].  Multiply Matrices")
        print("[2].  Inverse")
        print("[3].  Transpose")
        print("[4].  Determinant")
        print("[5].  Row reduce to echelon form")
        print("[6].  Basis for left and right nullspace")
        print("[7].  Rank and Basis for Row space and Column space")
        print("[8].  Eigenvalues and Eigenvectors")
        print("[9].  Diagonalize/Jordan Normal")
        print("[10]. Delete rows/cols")
        print("[11]. Applied Maths Mode")
        print("[12]. Calculus Mode: Differential Equations")
        print("[13]. Save/Load matrices from file")
        print("[14]. How To guide")
        print("")
        available_nums = list(range(15))
        available_nums.append(420) 
        while True:
            try:
                mode_choice = int(input("Mode No.: "))
                if mode_choice in available_nums:
                    return mode_choice
                else:
                    print("Not a valid mode number!")
                    pass

            except ValueError:
                print("")
                print("Not a valid choice!")
                print("")
                pass


    def sicko_mode(self):
        print("")
        print("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("")


    def row_or_col_del_mode(self):
        """ Loop for deleting rows or cols."""
        saved_matrices = self.saved_matrices
        print("Mode 10: Delete rows/cols")
        print("Warning: These deletions will overwrite saved matrices")
        A = self.input_matrix()
        print("")
        print("Input matrix:")
        self.dprint(A)
        while True:
            try:
                print("")
                print("Deletion Modes:")
                print("[0]. Back to main menu")
                print("[1]. Delete Row")
                print("[2]. Delete Column")
                row_or_col = int(input("Deletion Mode: "))
                if row_or_col == 1:
                    while True: # Row deletion loop
                        print("Row Deletion Mode")
                        print("")
                        row_del_num = int(input("Row number to delete: "))
                        try:
                            A.row_del(row_del_num - 1) # we -1 because starts from 0
                        except IndexError:
                            print("Row {} is not a valid row!".format(row_del_num))
                            print("No rows deleted!")
                        print("")
                        print("Result after deletion:")
                        self.dprint(A)
                        print("")
                        row_mode_choice = input("Delete another row? y/n: ")

                        if row_mode_choice == 'y':
                            continue

                        elif row_mode_choice == 'n':
                            self.save_matrix(A)
                            print("Result saved!")
                            print("")
                            break

                        else:
                            print("Not a valid input!")
                            continue

                        self.save_matrix(A)
                        print("Result saved!")
                        print("")
                        break

                elif row_or_col == 2:
                    while True: # Row deletion loop
                        print("Col Deletion Mode")
                        print("")
                        col_del_num = int(input("Col number to delete: "))
                        try:
                            A.col_del(col_del_num - 1) # we -1 because starts from 0
                        except IndexError:
                            print("Col {} is not a valid column!".format(col_del_num))
                            print("No columns deleted!")
                        print("")
                        print("Result after deletion:")
                        self.dprint(A)
                        print("")
                        col_mode_choice = input("Delete another column? y/n: ")

                        if col_mode_choice == 'y':
                            continue

                        elif col_mode_choice == 'n':
                            self.save_matrix(A)
                            print("Result saved!")
                            print("")
                            break

                        else:
                            print("Not a valid input!")
                            continue

                        break

                elif row_or_col == 0:
                    print("")
                    break

                else:
                    print("Not a valid input! Please input 1 or 2.")

            except ValueError:
                print("Not a valid input!")



    def applied_maths_mode(self):
        """Mode for various graph calculations based off incidence matrix"""
        saved_matrices = self.saved_matrices
        using_jupyter = self.using_jupyter
        while True:
            print("")
            print("Mode 11: Applied Maths Mode")
            print("[0]. Return to main menu")
            print("[1]. Electric Circuits: Solve for Ceff and potentials")
            print("[2]. Method of relaxation iterator")
            print("[3]. Spring Mass Systems: Solve for displacement and external forces")
            print("")
            applied_maths_mode_choice = input("Option: ")
            self.clear_previous()

            if applied_maths_mode_choice == '0':
                break

            elif applied_maths_mode_choice == '1':
                self.clear_previous()
                print("")
                print("Electric Circuits Solver")
                print("Incidence matrix:")
                A = self.input_matrix()
                print("")
                self.dprint(A)
                print("")
                A_row_num = A.shape[0]
                while True:
                    conductance_choice = input("Unit conductance? y/n: ")
                    if conductance_choice == 'y': # All edges have unit conductance
                        K = (A.T) * A
                        print("Laplacian K = A.T * A:")
                        self.dprint(K)
                        print("")

                    elif conductance_choice == 'n':
                        edge_weights = []
                        print("Please give edge weights")
                        for i in range(1, A_row_num + 1):
                            edge_weight = self.get_input("Weight {}, corresponding to row {}: ".format(i, i))
                            edge_weights.append(edge_weight)

                        C = zeros(A_row_num, A_row_num) # Matrix of edge weights
                        for i in range(len(edge_weights)):
                            C[i,i] = edge_weights[i]
                        print("")
                        print("Matrix of edge weights:")
                        self.dprint(C)
                        print("")
                        K = (A.T) * C * A # Weighted Laplacian
                        print("Weighted Laplacian K = A.T * C * A:")
                        self.dprint(K)
                        print("")

                    else:
                        print("Please input y/n!")
                        continue

                    IK = ImmutableMatrix(K) # Making sure our saved K doesn't get overwritten
                    self.save_matrix(IK)
                    K_row_num = IK.shape[0]
                    # So now we have our Laplacian K, next we get our vector of potentials
                    X = zeros(K_row_num, 1)
                    print("Vector of Node Potentials, x:")
                    print("")
                    print("Recommended Inputs: Use x_i, e.g x_3 for the unknown node potentials,")
                    print("1 for source and 0 for sink nodes")
                    for i in range(K_row_num):
                        X[i,0] = self.get_input("Node {} potential: ".format(i+1))
                    print("")
                    print("Vector of node potentials: ")
                    self.dprint(X)
                    IX = ImmutableMatrix(X) # Prevent overwriting 
                    # self.save_matrix(IX)
                    # So now we have our vector of node potentials X, next we get our vector of net flux divergence
                    f = zeros(K_row_num, 1)
                    print("Vector of net flux divergence from each node, f:")
                    print("")
                    print("Recommended Inputs: Use f for Ceff, 0 for interior nodes")
                    print("And anything you like for the sink nodes since they get deleted anyway")
                    for i in range(K_row_num):
                        f[i,0] = self.get_input("Node {} net flux divergence: ".format(i+1))
                    print("")
                    print("Vector of net flux divergences: ")
                    self.dprint(f)
                    If = ImmutableMatrix(f) # Prevent overwriting 
                    # self.save_matrix(If)
                    print("")
                    # Next we delete all rows and cols corresponding to zero potential, i.e grounded nodes
                    # First ascertain which node potentials are zero
                    grounded_row_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_row_idx)):
                        K.row_del(grounded_row_idx[i])
                        for i in range(i, len(grounded_row_idx)): # update indexes
                            grounded_row_idx[i] -= 1

                    grounded_col_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_col_idx)):
                        K.col_del(grounded_col_idx[i])
                        for i in range(i, len(grounded_col_idx)):
                            grounded_col_idx[i] -= 1

                    print("")
                    print("K grounded: ")
                    self.dprint(K)
                    print("")
                    # self.save_matrix(K)

                    # Next we have to ground f
                    grounded_flux_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_flux_idx)):
                        f.row_del(grounded_flux_idx[i])
                        for i in range(i, len(grounded_flux_idx)): # update indexes
                            grounded_flux_idx[i] -= 1

                    # print("f grounded: ")
                    # self.dprint(f)
                    # print("")

                    # Finally we ground X
                    grounded_potential_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_potential_idx)):
                        X.row_del(grounded_potential_idx[i])
                        for i in range(i, len(grounded_potential_idx)): # update indexes
                            grounded_potential_idx[i] -= 1

                    # print("x grounded: ")
                    # self.dprint(X)
                    # print("")

                    K_inv = K**-1
                    system = X - (K_inv * f) # = 0, System of equations
                    # variables = [type(x) for x in X]
                    variables = []

                    for x in list(set(X)|set(f)): # Union of lists
                        if isinstance(x, Symbol): # check if something is numerical or symbolic
                            variables.append(x)
                        # else do nothing
                    final_solutions = nonlinsolve(system, variables)
                    print("Solved Potentials/Ceff solution pairs:")
                    self.dprint([(variables[i], list(final_solutions)[0][i]) for i in range(len(variables))])
                    # self.dprint([(Matrix([variables])).T, '==', (Matrix([final_solutions])).T])
                    # variables_vector = (Matrix([variables])).T
                    # self.dprint(list(variables_vector))
                    # self.dprint(Matrix([final_solutions]).T)

                    break # Break out of applied maths mode

            elif applied_maths_mode_choice == '2':
                self.clear_previous()
                print("Method of relaxation iterator:")
                print("")
                A = self.input_adjacency()
                print("")
                print("Adjacency Matrix:")
                self.dprint(A)
                print("")
                num_nodes = A.shape[0]
                while True:
                    try:
                        print("Input Node numbers (space seperated):")
                        source_node_idx_string = input("Source Nodes: ").split(" ")
                        source_node_idx = [int(node) - 1 for node in source_node_idx_string]
                        sink_node_idx_string = input("Sink Nodes: ").split(" ")
                        sink_node_idx = [int(node) - 1 for node in sink_node_idx_string]
                        break
                    except ValueError:
                        print("Detected non-integer input!")

                boundary_node_idx = source_node_idx + sink_node_idx
                node_potentials = [0] * num_nodes
                for idx in source_node_idx:
                    node_potentials[idx] = 1

                while True:
                    try:
                        print("")
                        num_iterations = int(input("Number of iterations: "))
                        break
                    except ValueError:
                        print("INTEGER INTEGER INTEGER YOU MORON!")

                print("")
                print("Iterating...")
                self.harmonic_iterator(node_potentials, boundary_node_idx, A, num_iterations)
                print("")


            elif applied_maths_mode_choice == '3':
                self.clear_previous()
                print("")
                print("Spring Mass System Solver")
                print("")
                equilibrium_choice = input("Is the system in equilibrium? y/n: ")
                if equilibrium_choice == 'y':
                    print("")
                    print("Incidence matrix:")
                    A = self.input_matrix()
                    print("")
                    self.dprint(A)
                    print("")
                    A_row_num = A.shape[0]
                    while True:
                        edge_weights = []
                        print("Please give spring constants, (edge weights)")
                        for i in range(1, A_row_num + 1):
                            edge_weight = self.get_input("Spring constant {}, corresponding to row {}: ".format(i, i))
                            edge_weights.append(edge_weight)
                        C = zeros(A_row_num, A_row_num) # Matrix of edge weights
                        for i in range(len(edge_weights)):
                            C[i,i] = edge_weights[i]
                        print("")
                        print("Matrix of spring constants:")
                        self.dprint(C)
                        print("")
                        K = (A.T) * C * A # Weighted Laplacian
                        print("Weighted Laplacian K = A.T * C * A:")
                        self.dprint(K)
                        print("")
                        IK = ImmutableMatrix(K) # Making sure our saved K doesn't get overwritten
                        self.save_matrix(IK)
                        K_row_num = IK.shape[0]
                        # So now we have our Laplacian K, next we get our vector of potentials
                        X = zeros(K_row_num, 1)
                        print("Vector of Displacements \u03A6:")
                        print("")
                        print("Recommended Inputs: Use phi_2, e.g phi_3 for the mass displacements,")
                        print("0 for walls. (Note phi gets rendered as \u03C6)")
                        for i in range(K_row_num):
                            X[i,0] = self.get_input("Mass {} displacement: ".format(i+1))
                        print("")
                        print("Vector of mass displacements \u03A6: ")
                        self.dprint(X)
                        IX = ImmutableMatrix(X) # Prevent overwriting 
                        # self.save_matrix(IX)
                        # So now we have our vector of node potentials X, next we get our vector of net flux divergence
                        f = zeros(K_row_num, 1)
                        print("")
                        print("Vector of external forces, i.e [R_1, mg, mg, R_2]^T:")
                        print("")
                        print("Recommended Inputs: e.g mg for mass and R for reaction forces")
                        for i in range(K_row_num):
                            external_force = self.get_input("Node {} external force: ".format(i+1))
                            f[i,0] = external_force

                        print("")
                        print("Vector of external forces:")
                        self.dprint(f)
                        If = ImmutableMatrix(f) # Prevent overwriting 
                        # self.save_matrix(If)
                        print("")
                        # Next we delete all rows and cols corresponding to zero potential, i.e grounded nodes
                        # First ascertain which node potentials are zero
                        grounded_row_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                        for i in range(len(grounded_row_idx)):
                            K.row_del(grounded_row_idx[i])
                            for i in range(i, len(grounded_row_idx)): # update indexes
                                grounded_row_idx[i] -= 1

                        grounded_col_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                        for i in range(len(grounded_col_idx)):
                            K.col_del(grounded_col_idx[i])
                            for i in range(i, len(grounded_col_idx)):
                                grounded_col_idx[i] -= 1

                        print("K grounded: ")
                        self.dprint(K)
                        # Next we have to ground f
                        grounded_flux_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                        for i in range(len(grounded_flux_idx)):
                            f.row_del(grounded_flux_idx[i])
                            for i in range(i, len(grounded_flux_idx)): # update indexes
                                grounded_flux_idx[i] -= 1
                        # Finally we ground X
                        grounded_potential_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                        for i in range(len(grounded_potential_idx)):
                            X.row_del(grounded_potential_idx[i])
                            for i in range(i, len(grounded_potential_idx)): # update indexes
                                grounded_potential_idx[i] -= 1
                        K_inv = K**-1
                        system = X - (K_inv * f) # = 0, System of equations
                        # variables = [type(x) for x in X]
                        displacement_variables = []
                        for x in X:
                            if isinstance(x, Symbol): # check if something is numerical or symbolic
                                displacement_variables.append(x)
                        # First we solve for the displacements
                        # Tuple of solutions
                        displacement_solutions = list(nonlinsolve(system, displacement_variables))[0]
                        # Solving for internal forces
                        X_solved_grounded = (Matrix([displacement_solutions]).T)
                        internal_forces_variables = [Symbol(f'f_{i}') for i in range(1, len(X_solved_grounded) + 1)]
                        internal_forces = Matrix([internal_forces_variables]).T
                        system = K * X_solved_grounded + internal_forces 
                        internal_forces_solved = list(nonlinsolve(system, internal_forces_variables))[0]
                        # Regenerate our vector of mass displacements using our solutions 
                        X_solved_list = [0] * len(IX)
                        j = 0 # Counter for displacement solutions
                        for i in range(len(IX)):
                            if IX[i] == 0:
                                pass # Should be 0
                            else:
                                X_solved_list[i] = displacement_solutions[j]
                                j += 1

                        X_solved = (Matrix([X_solved_list])).T # List -> Matrix
                        # Solving for external forces
                        system = If - (IK * X_solved)
                        external_forces_variables = []
                        # This loop gathers all the reaction forces into a list of variables
                        for i in range(len(IX)):
                            if IX[i] == 0 and isinstance(If[i], Symbol): # Check 0 and If[i] is not a known value
                                external_forces_variables.append(If[i])

                        external_forces_solved = list(nonlinsolve(system, external_forces_variables))[0]
                        print("Mass displacements Solved:")
                        self.dprint([IX, X_solved])
                        print("")
                        print("External Forces Solved:")
                        self.dprint([Matrix([external_forces_variables]).T, Matrix([external_forces_solved]).T])
                        print("")
                        print("Internal Forces Solved:")
                        # self.dprint([internal_forces, '==', (Matrix([internal_forces_solved])).T])
                        self.dprint([internal_forces, Matrix([internal_forces_solved]).T])
                        print("")
                        # for i in range(len(variables)): # Print out final solutions
                        #     print("{} = {}".format(variables[i], list(final_solutions)[0][i]))
                        break # Break out of applied maths mode

                elif equilibrium_choice == 'n':
                    print("Newtons 2nd Law Problem:")
                    print("")
                    print("Incidence matrix:")
                    t = Symbol('t') # Define the time independent variable
                    A = self.input_matrix()
                    print("")
                    self.dprint(A)
                    print("")
                    A_row_num = A.shape[0]
                    edge_weights = []
                    print("Please give spring constants, (edge weights)")
                    for i in range(1, A_row_num + 1):
                        edge_weight = self.get_input("Spring constant {}, corresponding to row {}: ".format(i, i))
                        edge_weights.append(edge_weight)
                    C = zeros(A_row_num, A_row_num) # Matrix of edge weights
                    for i in range(len(edge_weights)):
                        C[i,i] = edge_weights[i]
                    print("")
                    print("Matrix of spring constants:")
                    self.dprint(C)
                    print("")
                    K = (A.T) * C * A # Weighted Laplacian
                    print("Weighted Laplacian K = A.T * C * A:")
                    self.dprint(K)
                    print("")
                    IK = ImmutableMatrix(K) # Making sure our saved K doesn't get overwritten
                    self.save_matrix(IK)
                    K_row_num = IK.shape[0]
                    # So now we have our Laplacian K, next we get our vector of potentials
                    X = zeros(K_row_num, 1)
                    print("Vector of Displacements \u03A6:")
                    print("")
                    print("Recommended Inputs: Use phi_2, e.g phi_3 for the mass displacements,")
                    print("0 for walls. (Note phi gets rendered as \u03C6)")
                    for i in range(K_row_num):
                        phi_string = input("Mass {} displacement: ".format(i+1))
                        if phi_string == '0':
                            phi = 0
                        else:
                            phi = Function(phi_string)(t)
                        X[i,0] = phi
                    print("")
                    print("Vector of mass displacements \u03A6: ")
                    self.dprint(X)
                    IX = ImmutableMatrix(X) # Prevent overwriting 
                    # self.save_matrix(IX)
                    # So now we have our vector of node potentials X, next we get our vector of net flux divergence
                    f = zeros(K_row_num, 1)
                    print("Vector of external forces, i.e [R_1, mg, mg, R_2]^T:")
                    print("")
                    print("Recommended Inputs: e.g mg for mass and R for reaction forces")
                    for i in range(K_row_num):
                        external_force = self.get_input("Node {} external force: ".format(i+1))
                        f[i,0] = external_force

                    print("")
                    print("Vector of external forces:")
                    self.dprint(f)
                    If = ImmutableMatrix(f) # Prevent overwriting 
                    # self.save_matrix(If)
                    print("")
                    # Next we delete all rows and cols corresponding to zero potential, i.e grounded nodes
                    # First ascertain which node potentials are zero
                    grounded_row_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_row_idx)):
                        K.row_del(grounded_row_idx[i])
                        for i in range(i, len(grounded_row_idx)): # update indexes
                            grounded_row_idx[i] -= 1

                    grounded_col_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_col_idx)):
                        K.col_del(grounded_col_idx[i])
                        for i in range(i, len(grounded_col_idx)):
                            grounded_col_idx[i] -= 1

                    print("K grounded: ")
                    self.dprint(K)
                    print("")
                    # self.save_matrix(K)

                    # Next we have to ground f
                    grounded_flux_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_flux_idx)):
                        f.row_del(grounded_flux_idx[i])
                        for i in range(i, len(grounded_flux_idx)): # update indexes
                            grounded_flux_idx[i] -= 1

                    # Finally we ground X
                    grounded_potential_idx = [i for i in range(K_row_num) if X[i,0] == 0]
                    for i in range(len(grounded_potential_idx)):
                        X.row_del(grounded_potential_idx[i])
                        for i in range(i, len(grounded_potential_idx)): # update indexes
                            grounded_potential_idx[i] -= 1


                    while True:
                        unit_mass_choice = input('Unit masses y/n: ')
                        if unit_mass_choice == 'y' or unit_mass_choice == 'n':
                            break
                        else:
                            print("Not a valid input!")
                            continue

                    if unit_mass_choice == 'y':
                        M = eye(K.shape[0])
                    else:
                        masses = []
                        print("Please give masses: ")
                        for i in range(1, K.shape[0] + 1):
                            mass = self.get_input("Mass {}, corresponding to row {}: ".format(i, i))
                            masses.append(mass)

                        M = zeros(K.shape[0], K.shape[0]) # Grounded M
                        for i in range(len(masses)):
                            M[i,i] = masses[i]
                            
                    eigen_pairs = K.eigenvects()
                    print("")
                    print("If \u03BB is an eigen value then \u03C9, s.t \u03C9\u00B2 = \u03BB is a 'natural frequency'.")
                    print("Eigenvalue, natural frequency pairs (\u03BB, \u03C9) for the system:")
                    self.dprint([(eigen_pair[0], sqrt(eigen_pair[0])) for eigen_pair in eigen_pairs])
                    print("")
                    print("Corresponding Eigenvectors/Natural Modes of Oscillation:")
                    # self.dprint([eigen_pair[2] for eigen_pair in eigen_pairs])
                    self.print_eigen(K)
                    print("")
                    print("Warning: This next part could hang if the solutions are not easily found.")
                    second_derivative_phi = diff(diff(X, t), t)
                    system = M*second_derivative_phi + K*X - f
                    solutions = dsolve(system, [phi for phi in X])
                    print("Displacement solutions in cos and sin: ")
                    for solution in solutions:
                        self.dprint(solution)
                    print("")
                    print("Displacement solutions in exp: ")
                    for solution in solutions:
                        self.dprint(solution.rewrite(exp).simplify())
                    print("")
                    print("Displacement solutions expanded: ")
                    for solution in solutions:
                        self.dprint(solution.rewrite(exp).simplify().expand())

            else:
                print("Not a valid input!")
                continue


    def print_phase_warning(self):
        print("""
The above is a screenshot from the notes to give a general idea of the
phase portrait in this (trace, det) section. Below is an actual plot
of the vector field and the lines given by the eigen vectors.
(Note: no eigenvector lines for complex eigenvectors).
Combined it is easy to see what the phase portrait should be.
        """)

    def jupyter_phase_portrait_printer(self, A):
        """
        If using jupyter version, display the phase portrait general shape,
        from images.
        Arguments: jupyter: bool, A: Square Sympy Matrix
        """
        using_jupyter = self.using_jupyter
        if A.shape[0] == 2 and A.shape[1] == 2: # if we have a 2x2 matrix
            try:
                A_diagonalizable = True
                if using_jupyter: 
                    try:
                        P, D = A.diagonalize()
                    except MatrixError: # Not diagonalizable
                        A_diagonalizable = False
                    trace_A = A.trace()
                    det_A = A.det()
                    trace24 = (trace_A**2) / 4 # tau^2/4
                    print("")
                    print("Trace: {}".format(trace_A))
                    print("Determinant: {}".format(det_A))
                    print("")
                    if det_A < 0:
                        print("")
                        display(Math(r'\Delta < 0'))
                        print("Saddle-point or Hyperbolic profile:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_1.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A > 0 and det_A < trace24 and trace_A > 0:
                        print("")
                        display(Math(r'\Delta > 0, ~ ~ \Delta < \frac{\tau^2}{4}, ~ ~ \tau > 0'))
                        print("Repelling or unstable node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_2.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A > 0 and det_A < trace24 and trace_A < 0:
                        print("")
                        display(Math(r'\Delta > 0, ~ ~ \Delta < \frac{\tau^2}{4}, ~ ~ \tau < 0'))
                        print("Attracting or stable node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_3.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A > 0 and trace_A == 0:
                        print("")
                        display(Math(r'\Delta > 0, ~ ~ \tau = 0'))
                        print("Centre or elliptic profile:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_4.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A > trace24 and trace_A > 0:
                        print("")
                        display(Math(r'\Delta > \frac{\tau^2}{4}, ~ ~ \tau > 0'))
                        print("Repelling or unstable spiral:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_5.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A > trace24 and trace_A < 0:
                        print("")
                        display(Math(r'\Delta > \frac{\tau^2}{4}, ~ ~ \tau < 0'))
                        print("Attracting or stable spiral:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_6.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A == 0 and trace_A > 0:
                        print("")
                        display(Math(r'\Delta = 0, ~ ~ \tau > 0'))
                        print("Line of repelling or unstable fixed points.")
                        display(Image(filename="Images/phase_portraits/phase_portrait_7.png"))
                        self.print_phase_warning()
                        print("")

                    elif det_A == 0 and trace_A < 0:
                        print("")
                        display(Math(r'\Delta = 0, ~ ~ \tau < 0'))
                        print("Line of attracting or stable fixed points.")
                        display(Image(filename="Images/phase_portraits/phase_portrait_8.png"))
                        self.print_phase_warning()
                        print("")

                    elif trace_A**2 - 4*det_A == 0 and trace_A > 0 and A_diagonalizable:
                        print("")
                        display(Math(r'\tau^2 - 4\Delta = 0, ~ ~ \tau > 0, ~ ~ \text{A Diagonalizable}'))
                        print("Repelling Star Node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_9.png"))
                        self.print_phase_warning()
                        print("")

                    elif trace_A**2 - 4*det_A == 0 and trace_A < 0 and A_diagonalizable:
                        print("")
                        display(Math(r'\tau^2 - 4\Delta = 0, ~ ~ \tau < 0, ~ ~ \text{A Diagonalizable}'))
                        print("Attracting Star Node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_10.png"))
                        self.print_phase_warning()
                        print("")

                    elif trace_A**2 - 4*det_A == 0 and trace_A > 0 and not A_diagonalizable:
                        print("")
                        display(Math(r'\tau^2 - 4\Delta = 0, ~ ~ \tau > 0, ~ ~ \text{A Not Diagonalizable}'))
                        print("Unstable Degenerate Node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_11.png"))
                        self.print_phase_warning()
                        print("")

                    elif trace_A**2 - 4*det_A == 0 and trace_A < 0 and not A_diagonalizable:
                        print("")
                        display(Math(r'\tau^2 - 4\Delta = 0, ~ ~ \tau < 0, ~ ~ \text{A Not Diagonalizable}'))
                        print("Stable Degenerate Node:")
                        display(Image(filename="Images/phase_portraits/phase_portrait_12.png"))
                        self.print_phase_warning()
                        print("")

                    else:
                        pass # Do nothing if not using jupyter

            except TypeError:
                pass

        else: # If A is not a 2x2 we don't want to print anything
            pass

    def plot_vf(self, A, x_range=[-5, 5], y_range=[-5, 5], grid=[21, 21]):
        """
        Plot vector field for a 2x2 matrix system of ODEs.
        Also Plots the lines corresponding to eigenvectors.
        Args: A == 2x2 Sympy Matrix
        Args: x_range == xlim for plot
        Args: y_range == ylim for plot
        Args: grid == grid size to plot
        # Inspiration: https://gist.github.com/nicoguaro/6767643
        """
        if not A.is_square or A.shape[0] != 2: # if A is not 2x2
            return None

        else:
            try:
                x = np.linspace(x_range[0], x_range[1], grid[0])
                y = np.linspace(y_range[0], y_range[1], grid[1])
                X , Y  = np.meshgrid(x, y)  # create a grid
                dxdt = float(A[0,0])*X + float(A[0,1])*Y  # Multiplying out A
                dydt = float(A[1,0])*X + float(A[1,1])*Y 
                norm = np.hypot(dxdt, dydt)  # Length of each arrow
                norm[norm == 0] = 1.0        # Avoid zero division errors 
                dxdt = dxdt / norm           # Normalize each arrow
                dydt = dydt / norm
                # Calculate eigenvectors
                eigen_triples = A.eigenvects()
                v1 = eigen_triples[0][2][0] 
                if len(eigen_triples) == 1: # repeat root
                    if len(eigen_triples[0][2]) == 1: # dim espace == 1
                        v2 = v1
                    else: # eigenspace has two basis
                        v2 = eigen_triples[0][2][1] 
                else: # non-repeat evalue
                    v2 = eigen_triples[1][2][0]
                if (v1[0].is_real and v1[1].is_real and
                    v2[0].is_real and v2[1].is_real):
                    # Plotting eigenvectors, don't plot if complex valued.
                    plt.plot(float(v1[0])*x,float(v1[1])*y, '-r', color='r') 
                    plt.plot(float(v2[0])*x,float(v2[1])*y, '-r', color='r') 
                # Plotting vector field
                plt.quiver(X, Y, dxdt, dydt, pivot='mid', color='k')
                plt.axis('equal')
                plt.xlim(x_range), plt.ylim(y_range)
                plt.grid('on')
                plt.show()

            except TypeError: # Symbolic inputs
                print("Non-numerical input => can't plot vector field!")
                return None

    def get_ode(self, y_string, x_string):
        """
        Gets input for Order and Coefficients,
        Returns: tuple, (homogenous part - non-homogenous part, homogenous part).
        Args: y_string == Dependent Variable as a string, e.g 'y'
        Args: x_string == Independent Variable as a string, e.g 'x'
        """
        using_jupyter = self.using_jupyter
        while True: # Input Loop
            if using_jupyter:
                print("ODE of the form:")
                display(Math(r"$$ \alpha_0y + \alpha_1\frac{dy}{dx} + \ldots"
                            r"+ \alpha_{n-1}\frac{d^{n-1}y}{dx^{n-1}} + \alpha_n\frac{d^ny}{dx^n} = RHS$$"))
                print("")
            else:
                print(f"A*d(n){y_string}/d{x_string}(n) + B*d(n-1){y_string}/d{x_string}(n-1)"
                    f"+ ... + C*d(1){y_string}/d{x_string}(1) + D*{y_string} = RHS")
            try:
                order = int(input("Order, n: "))
                break
            except ValueError:
                print("Please input a a positive integer!")
                continue

        x = Symbol(x_string)
        y = Function(y_string)(x)
        homo_string = input(f"{y_string} coefficient: ") + f'*{y_string}({x_string}) + '
        for i in range(1, order):
            coefficient = input(f"d{i}{y_string}/d{x_string}{i} coefficient: ")
            homo_string += coefficient + '*' + f'Derivative({y_string}({x_string}), ({x_string}, {i})) + '

        coefficient = input(f"d{order}{y_string}/d{x_string}{order} coefficient: ")
        homo_string += coefficient + '*' + f'Derivative({y_string}({x_string}), ({x_string}, {order}))'
        no_homo_string = " - {}".format(input("RHS: "))
        return (sympify(homo_string), sympify(homo_string + no_homo_string))


    def calculus_mode(self):
        """Calculus and applications mode"""
        using_jupyter = self.using_jupyter
        print("")
        print("Calculus Mode")
        while True:
            print("")
            print("Options:")
            print("[0]. Return to main menu")
            print("[1]. System of ODEs Solver")
            print("[2]. Plot phase portrait for a 2x2 matrix system.")
            print("[3]. Singular ODE Solver")
            print("")
            calculus_mode_choice = input("Option: ")
            self.clear_previous()
            if calculus_mode_choice == '0':
                break

            elif calculus_mode_choice == '1':
                print("")
                print("System of ODEs Solver: dY/dt = A*Y + g(t)")
                print("")
                t = Symbol('{}'.format(input("Independent variable, i.e t: ")))
                print("")
                print("Matrix A:")
                A = self.input_matrix()
                A_row_num = A.shape[0]
                print("")
                self.dprint(A)
                print("")
                print("Input dependent variables, i.e x, y, z:")
                Y_list = []
                for i in range(A_row_num): # get vector of variables
                    variable_string = input("Variable {}: ".format(i+1))
                    variable = Function(variable_string)(t) # e.g convert 'x' to x(t)
                    Y_list.append(variable)
                print("")
                print("Non-homogenous part vector g(t):")
                g_list = []
                for i in range(A_row_num): # Get the non-homogeneous part
                    g_element = sympify(input("g(t) row {} entry: ".format(i+1))) # RHS = g(t)
                    g_list.append(g_element)

                Y = Matrix([Y_list]).T # converting lists to vectors
                g = Matrix([g_list]).T
                dYdt = diff(Y, t)
                system = dYdt - (A*Y + g)
                solutions = dsolve(system, Y_list)
                print("")
                print("Warning: This could hang if the eigenvectors are not easily found.")
                self.print_eigen(A)
                self.diagonalize_or_jordanize(A)
                print("Solutions:")
                self.dprint(self.simplify_matrix(Matrix([solutions]).T))
                print("")
                self.jupyter_phase_portrait_printer(A)
                print("")
                print("Plot of the actual vector field and eigenvector lines:")
                self.plot_vf(A)
                print("")
                continue

            elif calculus_mode_choice == '2':
                print("Phase Portrait Plotter for 2x2 matrix systems")
                while True:
                    A = self.input_matrix()
                    if A.shape[0] != 2 or A.shape[1] != 2:
                        print("Only accepts 2x2 matrices!")
                        continue
                    else:
                        break
                print("")
                self.jupyter_phase_portrait_printer(A)
                print("")
                print("Plot of the actual vector field and eigenvector lines:")
                self.plot_vf(A)
                print("")
                continue


            elif calculus_mode_choice == '3':
                print("Differential Equation solver")
                y_string = input("Input dependent variable, i.e y for dy/dx: ")
                x_string = input("Input independent variable, i.e x for dy/dx: ")
                x = Symbol(x_string)
                y = Function(y_string)(x)
                solutions = self.get_ode(y_string, x_string)
                print("")
                print("Complimentary general solution:")
                self.dprint(dsolve(solutions[0], y))
                print("Full general solution with non-homogenous part:")
                self.dprint(dsolve(solutions[1], y))

            else:
                print("Not a valid input!")
                continue


    def save_or_load_mode(self):
        """Save currently saved matrices to file or load from file."""
        saved_matrices = self.saved_matrices
        while True:
            print("")
            print("Save/Load Mode:")
            print("[0]. Back to main menu")
            print("[1]. Save currently saved matrices to memory.txt")
            print("[2]. Load saved matrices from memory.txt")
            print("[3]. Erase saved matrices")
            print("[4]. Print all currently saved matrices")
            print("")
            save_or_load = input("Option: ")
            if save_or_load == '0':
                break # back to main menu

            elif save_or_load == '1':
                print("Saving {} matrices to memory.txt...".format(len(saved_matrices)))
                try:
                    with open('memory.txt', 'rb') as f:
                        # Load already saved
                        file_matrices = pickle.load(f)

                    saved_matrices += [matrix for matrix in file_matrices if matrix not in saved_matrices]
                except EOFError: # don't append to saved_matrices if no saved matrices exit
                    pass

                with open('memory.txt', 'wb') as f:
                    pickle.dump(saved_matrices, f)

                print("Successfully saved to memory.txt!")
                continue

            elif save_or_load == '2':
                print("Loading matrices from memory.txt...")
                try:
                    try:
                        with open('memory.txt', 'rb') as f:
                            file_matrices = pickle.load(f)
                            saved_matrices += [matrix for matrix in file_matrices if matrix not in saved_matrices]

                        print("Successfully loaded {} matrices from memory.txt".format(len(file_matrices)))
                        continue
                    except FileNotFoundError:
                        print("memory.txt not found, please create.")

                except EOFError:
                    print("No saved matrices to load! Save some first!") 

                continue

            elif save_or_load == '3':
                print("Erasing contents of memory.txt and resetting list of saved matrices...")
                try:
                    open('memory.txt', 'w').close()
                    print("Successfully erased all saved matrices!")
                    self.saved_matrices = []

                except FileNotFoundError:
                    print("memory.txt not found, please create.")

            elif save_or_load == '4':
                self.print_saved_matrices()

            else:
                print("Not a valid input!")
                continue


def main(using_jupyter):
    """Main Input Loop."""
    saved_matrices = [] # initialize
    # Initiate Solvers object
    solver = Solvers(using_jupyter=using_jupyter, saved_matrices=saved_matrices)
    title_text = """
___  ___      _        _        _____       _                
|  \/  |     | |      (_)      /  ___|     | |               
| .  . | __ _| |_ _ __ ___  __ \ `--.  ___ | |_   _____ _ __ 
| |\/| |/ _` | __| '__| \ \/ /  `--. \/ _ \| \ \ / / _ \ '__|
| |  | | (_| | |_| |  | |>  <  /\__/ / (_) | |\ V /  __/ |   
\_|  |_/\__,_|\__|_|  |_/_/\_\ \____/ \___/|_| \_/ \___|_|   
                                                             
Author: Isaac Lee

"""
    try:
        try:
            with open('memory.txt', 'rb') as f: # load saved matrices
                file_matrices = pickle.load(f)
                saved_matrices += [matrix for matrix in file_matrices if matrix not in saved_matrices]
        except FileNotFoundError:
            pass


    except EOFError:
        pass

    while True:
        solver.clear_previous()
        print(title_text)
        mode_num = solver.mode_selector()
        try:
            if mode_num == 0:
                print("")
                print("Bye!")
                print("")
                break

            if mode_num == 1:
                solver.clear_previous()
                while True:
                    print("Matrix Multiplication Mode")
                    print("")
                    print("[0]. Return to main menu")
                    print("[1]. Product: AB")
                    print("[2]. Power: A^n")
                    print("")
                    multiplication_mode_num = input("Option: ")

                    if multiplication_mode_num == '0':
                        break

                    if multiplication_mode_num == '1':
                        print("Input matrix A:")
                        A = solver.input_matrix()
                        solver.dprint(A)
                        print("")
                        print("Input Matrix B:")
                        B = solver.input_matrix() # get matrix B and save it
                        solver.dprint(B)
                        try:
                            product = A*B
                            print("")
                            print("A*B: ")
                            solver.dprint(product)
                            solver.save_matrix(product) 
                            print("")
                            print("Result Saved!")
                            print("")

                        except ShapeError:
                            print("")
                            print("Matrix Dimension Error!")
                            print("")
                        
                        continue

                    elif multiplication_mode_num == '2':
                        print("Input matrix A:")
                        A = solver.input_matrix()
                        solver.dprint(A)
                        print("")
                        n = solver.get_input("Input power, n: ")
                        try:
                            power = A**n
                            print("")
                            print("A^n: ")
                            solver.dprint(power)
                            solver.save_matrix(power) 
                            print("")
                            print("Result Saved!")
                            print("")

                        except ShapeError:
                            print("")
                            print("Matrix Dimension Error!")
                            print("")

                        continue

                    else:
                        print("Not a valid input!")
                        continue


            elif mode_num == 2:
                solver.clear_previous()
                print("Mode {}: Inverse".format(mode_num))
                A = solver.input_matrix()
                print("Input matrix:")
                solver.dprint(A)
                print("")
                print("A inverse:")
                try:
                    if A.det() != 0: 
                        A_inv = A**-1
                        solver.dprint(A_inv)
                        solver.save_matrix(A_inv)
                        print("")
                        print("Result Saved!")
                        print("")
                        main_menu_pause = input("Press any key to return to main menu: ")
                    else:
                        print("")
                        print("det = 0 => Not invertible!")
                        print("")
                        main_menu_pause = input("Press any key to return to main menu: ")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => Not invertible!")
                    print("")
                    main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 3:
                solver.clear_previous()
                print("Mode {}: Transpose".format(mode_num))
                A = solver.input_matrix()
                print("Input matrix:")
                solver.dprint(A)
                A_transpose = A.T
                print("")
                print("A transpose: ")
                solver.dprint(A_transpose)
                print("")
                solver.save_matrix(A_transpose)
                print("")
                print("Result Saved!")
                print("")
                main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 4:
                solver.clear_previous()
                print("Mode {}: Determinant".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                try:
                    A_det = A.det()
                    print("Determinant of A:")
                    solver.dprint(A_det)
                    print("")
                    main_menu_pause = input("Press any key to return to main menu: ")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => Det doesn't exist!")
                    print("")
                    main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 5:
                solver.clear_previous()
                print("Mode {}: Row reduced Echelon Form".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                A_row_reduced = A.rref()[0]
                print("A row reduced:")
                solver.dprint(A_row_reduced)
                print("")
                solver.save_matrix(A_row_reduced)
                if using_jupyter:
                    display(Image(filename="Images/basis_change.jpg"))
                print("Result Saved!")
                print("")
                main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 6:
                solver.clear_previous()
                print("Mode {}: Basis for left and right nullspace".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                print("Basis for Right Nullspace: ")
                solver.dprint(A.nullspace())
                print("")
                A_transpose = A.T
                print("Basis for Left Nullspace: ")
                solver.dprint(A_transpose.nullspace())
                print("")
                print("Note: If these are empty, the nullspace is just")
                print("the trivial zero vector.")
                print("")
                main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 7:
                solver.clear_previous()
                print("Mode {}: Rank and basis for Row and Col space".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                print("Basis for column space: ")
                solver.dprint(A.columnspace())
                print("")
                print("Basis for row space: ")
                solver.dprint(A.rowspace())
                print("")
                print("Rank: {}".format(len(A.columnspace())))
                print("")
                main_menu_pause = input("Press any key to return to main menu: ")


            elif mode_num == 8:
                solver.clear_previous()
                print("Mode {}: Eigenvalues and Eigenvectors".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                try:
                    x = symbols('x')
                    char_pol = A.charpoly(x)
                    print("Characteristic Polynomial")
                    solver.dprint(factor(char_pol.as_expr()))
                    print("")
                    print("Warning: if this is taking a long time, then it's likely there are no rational")
                    print("non-zero eigenvectors and you should just terminate the process.")
                    # solver.dprint(A.eigenvects())
                    solver.print_eigen(A)
                    key_to_continue = input("Press enter to return to main menu...")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => No eigenvectors or values!")
                    print("")
                    main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 9:
                solver.clear_previous()
                print("Mode {}: Diagonalize/Jordan Normal".format(mode_num))
                A = solver.input_matrix()
                print("")
                print("Input matrix:")
                solver.dprint(A)
                print("")
                solver.diagonalize_or_jordanize(A)
                main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 10:
                solver.clear_previous()
                solver.row_or_col_del_mode()

            elif mode_num == 11:
                solver.clear_previous()
                solver.applied_maths_mode()

            elif mode_num == 12:
                solver.clear_previous()
                solver.calculus_mode()

            elif mode_num == 13:
                solver.clear_previous()
                solver.save_or_load_mode()

            elif mode_num == 14:
                solver.clear_previous()
                with open('README.md', 'r') as f: # Print out contents of README.md
                    print(f.read())
                main_menu_pause = input("Press any key to return to main menu: ")

            elif mode_num == 420:
                solver.sicko_mode()
                break

        except KeyboardInterrupt:
            print("")
            print("Returning to main menu...")
            continue


if __name__ == "__main__":
    print("")
    print("Don't run this file...")
    print("Instead run matrix_solver.py or matrix_solver_jupyter.ipynb.")
    print("")

