from sympy import *
from sympy.matrices.common import MatrixError
from fractions import Fraction
import pickle
init_printing()

saved_matrices = []

def save_matrix(matrix_to_save):
    """ Save a matrix. """
    global saved_matrices
    if matrix_to_save not in saved_matrices:
        saved_matrices.append(matrix_to_save)
        print("")


def use_saved_matrices():
    """ Ask whether to use a saved matrix. """
    print("")
    use_saved = input("Use a saved matrix? y/n ")
    if use_saved == 'y':
        return True
    else:
        return False
    

def print_saved_matrices():
    """ 
    Print out available saved matrices along with their indexes.
    Also returns False if no matrices are saved and True if atleast
    1 saved matrix exists.
    """
    global saved_matrices
    if not saved_matrices:
        print("No saved matrices!")
        return False
    else:
        saved_with_indexes = [([i], saved_matrices[i]) for i in range(len(saved_matrices)) ]
        pprint(saved_with_indexes)
        return True


def ask_to_quit():
    """ 
    Ask a user if they want to quit.
    Returns True if they answer y
    Returns False if they answer anything else
    """
    quit_input = input("Quit: y/n ")
    if quit_input == 'y':
        return True
    else:
        return False
        

def get_input(prompt_text):
    """
    Takes both numerical and variable input 
    and returns the result as either a float or 
    a sympy symbol.
    Note: Also can take fractional input, e.g '1/2'.
    Note: For decimal input use fractions, if a float is given the
    integer part is used.
    """
    input_element = input(prompt_text)
    try:
        if '/' in [char for char in input_element]: # Check for fractions
            numerator = int(input_element.split('/')[0])
            denominator = int(input_element.split('/')[1])
            element = Fraction(numerator, denominator)

        else: # Not a fraction, integer inputs
            element = int(input_element.split('.')[0]) # Integer part if float

    except ValueError: # if we have an unknown letter input e.g x
        element = Symbol('{}'.format(input_element))

    return element


def input_matrix():
    """ 
    Either use a saved matrix or prompt for a new one.
    Returns the  matrix A.
    Returns None if an invalid saved matrix index is given.
    """
    global saved_matrices # this is lazy and bad but cba
    while True:
        if use_saved_matrices() and print_saved_matrices():
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
                R = int(input("No. Rows: ")) 
                C = int(input("No. Cols: "))
                A = Matrix([])
                for i in range(R):
                    ith_row = []
                    for j in range(C):
                        element = get_input("({}, {}) entry: ".format(i+1, j+1))
                        ith_row.append(element)
                        pprint(Matrix([ith_row]))
                    A = A.row_insert(i, Matrix([ith_row]))

                save_matrix(A)
                return A

            except ValueError:
                print("Not a valid input!")
                continue
                # Returns None


def mode_selector():
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
    print("[11]. Solve for Ceff (Applied Maths)")
    print("[12]. Save/Load matrices from file")
    print("")
    available_nums = list(range(13))
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


def sicko_mode():
    print("")
    print("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("")


def row_or_col_del_mode():
    """ Loop for deleting rows or cols."""
    global saved_matrices
    print("Mode 10: Delete rows/cols")
    print("Warning: These deletions will overwrite saved matrices")
    A = input_matrix()
    print("")
    print("Input matrix:")
    pprint(A)
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
                    pprint(A)
                    print("")
                    row_mode_choice = input("Delete another row? y/n: ")

                    if row_mode_choice == 'y':
                        continue

                    elif row_mode_choice == 'n':
                        save_matrix(A)
                        print("Result saved!")
                        print("")
                        break

                    else:
                        print("Not a valid input!")
                        continue

                    save_matrix(A)
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
                    pprint(A)
                    print("")
                    col_mode_choice = input("Delete another column? y/n: ")

                    if col_mode_choice == 'y':
                        continue

                    elif col_mode_choice == 'n':
                        save_matrix(A)
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



def applied_maths_mode():
    """Mode for various graph calculations based off incidence matrix"""
    print("Mode 11: Applied Maths Mode")
    global saved_matrices
    print("Incidence matrix:")
    A = input_matrix()
    print("")
    # For testing purposes
    # A = Matrix([[-1, 0, 1, 0, 0, 0],
    #             [0, 0, -1, 0, 0, 1],
    #             [0, 0, 0, 0, 1, -1],
    #             [0, 0, -1, 0, 1, 0],
    #             [0, 1, -1, 0, 0, 0],
    #             [1, -1, 0, 0, 0, 0],
    #             [0, 1, 0, 0, -1, 0],
    #             [0, 1, 0, -1, 0, 0],
    #             [0, 0, 0, 1, -1, 0]])
    pprint(A)
    print("")
    A_row_num = A.shape[0]
    while True:
        conductance_choice = input("Unit conductance? y/n: ")
        if conductance_choice == 'y': # All edges have unit conductance
            K = (A.T) * A
            print("Laplacian K = A.T * A:")
            pprint(K)
            print("")

        elif conductance_choice == 'n':
            edge_weights = []
            print("Please give edge weights")
            for i in range(1, A_row_num + 1):
                edge_weight = get_input("Weight {}, corresponding to row {}: ".format(i, i))
                edge_weights.append(edge_weight)

            C = zeros(A_row_num, A_row_num) # Matrix of edge weights
            for i in range(len(edge_weights)):
                C[i,i] = edge_weights[i]
            print("")
            print("Matrix of edge weights:")
            pprint(C)
            print("")
            K = (A.T) * C * A # Weighted Laplacian
            print("Weighted Laplacian K = A.T * C * A:")
            pprint(K)
            print("")

        else:
            print("Please input y/n!")
            continue
                
        IK = ImmutableMatrix(K) # Making sure our saved K doesn't get overwritten
        save_matrix(IK)
        K_row_num = IK.shape[0]
        # So now we have our Laplacian K, next we get our vector of potentials
        X = zeros(K_row_num, 1)
        print("Vector of Node Potentials, x:")
        print("")
        print("Recommended Inputs: Use x_i, e.g x_3 for the unknown node potentials,")
        print("1 for source and 0 for sink nodes")
        for i in range(K_row_num):
            X[i,0] = get_input("Node {} potential: ".format(i+1))
        print("")
        print("Vector of node potentials: ")
        pprint(X)
        IX = ImmutableMatrix(X) # Prevent overwriting 
        save_matrix(IX)
        # So now we have our vector of node potentials X, next we get our vector of net flux divergence
        f = zeros(K_row_num, 1)
        print("Vector of net flux divergence from each node, f:")
        print("")
        print("Recommended Inputs: Use f for Ceff, 0 for interior nodes")
        print("And anything you like for the sink nodes since they get deleted anyway")
        for i in range(K_row_num):
            f[i,0] = get_input("Node {} net flux divergence: ".format(i+1))
        print("")
        print("Vector of net flux divergences: ")
        pprint(f)
        If = ImmutableMatrix(f) # Prevent overwriting 
        save_matrix(If)
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
        pprint(K)
        save_matrix(K)

        # Next we have to ground f
        grounded_flux_idx = [i for i in range(K_row_num) if X[i,0] == 0]
        for i in range(len(grounded_flux_idx)):
            f.row_del(grounded_flux_idx[i])
            for i in range(i, len(grounded_flux_idx)): # update indexes
                grounded_flux_idx[i] -= 1

        print("f grounded: ")
        pprint(f)
        save_matrix(f)

        # Finally we ground X
        grounded_potential_idx = [i for i in range(K_row_num) if X[i,0] == 0]
        for i in range(len(grounded_potential_idx)):
            X.row_del(grounded_potential_idx[i])
            for i in range(i, len(grounded_potential_idx)): # update indexes
                grounded_potential_idx[i] -= 1
            
        print("x grounded: ")
        pprint(X)
        save_matrix(X)

        K_inv = K**-1
        system = X - (K_inv * f) # = 0, System of equations
        # variables = [type(x) for x in X]
        variables = []
        
        for x in list(set(X)|set(f)): # Union of lists
            if isinstance(x, Symbol): # check if something is numerical or symbolic
                variables.append(x)
            # else do nothing
        final_solutions = nonlinsolve(system, variables)
        for i in range(len(variables)): # Print out final solutions
            print("{} = {}".format(variables[i], list(final_solutions)[0][i]))

        break # Break out of applied maths mode

            
            
def save_or_load_mode():
    """Save currently saved matrices to file or load from file."""
    global saved_matrices
    while True:
        print("")
        print("Save/Load Mode:")
        print("[0]. Back to main menu")
        print("[1]. Save currently saved matrices to memory.txt")
        print("[2]. Load saved matrices from memory.txt")
        print("[3]. Erase contents of memory.txt")
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
                with open('memory.txt', 'rb') as f:
                    file_matrices = pickle.load(f)
                    saved_matrices += [matrix for matrix in file_matrices if matrix not in saved_matrices]

                print("Successfully loaded {} matrices from memory.txt".format(len(file_matrices)))
                continue

            except EOFError:
                print("No saved matrices to load! Save some first!") 

            continue

        elif save_or_load == '3':
            print("Erasing contents of memory.txt...")
            open('memory.txt', 'w').close()
            print("Successfully erased contents of memory.txt!")

        elif save_or_load == '4':
            print_saved_matrices()

        else:
            print("Not a valid input!")
            continue
    

def main():
    global saved_matrices
    intro_text = """
___  ___      _        _        _____       _                
|  \/  |     | |      (_)      /  ___|     | |               
| .  . | __ _| |_ _ __ ___  __ \ `--.  ___ | |_   _____ _ __ 
| |\/| |/ _` | __| '__| \ \/ /  `--. \/ _ \| \ \ / / _ \ '__|
| |  | | (_| | |_| |  | |>  <  /\__/ / (_) | |\ V /  __/ |   
\_|  |_/\__,_|\__|_|  |_/_/\_\ \____/ \___/|_| \_/ \___|_|   
                                                             
Author: Isaac Lee

IMPORTANT:

Inputs:
Matrix can take integer, fractional and symbolic inputs. 
E.g 1, 2, 3, 1/2, 11/12, a, b, c, pi, alpha, beta, gamma, delta, e.t.c. 
(Which get rendered using unicode). 

Because of how floats work in python I have opted to not allow decimal input,
any decimal input will be truncated to it's integer part. If you want a decimal input,
i.e 1.2, use a fractional input instead, i.e 6/5.
    
To return to main menu at any point use control c 
"""
    print(intro_text)

    print("Loading matrices from memory.txt...")
    try:
        with open('memory.txt', 'rb') as f:
            file_matrices = pickle.load(f)
            saved_matrices += [matrix for matrix in file_matrices if matrix not in saved_matrices]

        print("Successfully loaded {} matrices from memory.txt".format(len(file_matrices)))

    except EOFError:
        pass

    while True:
        mode_num = mode_selector()
        try:
            if mode_num == 0:
                print("")
                print("Bye!")
                print("")
                break

            if mode_num == 1:
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
                        A = input_matrix()
                        pprint(A)
                        print("")
                        print("Input Matrix B:")
                        B = input_matrix() # get matrix B and save it
                        pprint(B)
                        try:
                            product = A*B
                            print("")
                            print("A*B: ")
                            pprint(product)
                            save_matrix(product) 
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
                        A = input_matrix()
                        pprint(A)
                        print("")
                        n = get_input("Input power, n: ")
                        try:
                            power = A**n
                            print("")
                            print("A^n: ")
                            pprint(power)
                            save_matrix(power) 
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
                print("Mode {}: Inverse".format(mode_num))
                A = input_matrix()
                print("Input matrix:")
                pprint(A)
                print("")
                print("A inverse:")
                try:
                    if A.det() != 0: 
                        A_inv = A**-1
                        pprint(A_inv)
                        save_matrix(A_inv)
                        print("")
                        print("Result Saved!")
                        print("")
                    else:
                        print("")
                        print("det = 0 => Not invertible!")
                        print("")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => Not invertible!")
                    print("")

            elif mode_num == 3:
                print("Mode {}: Transpose".format(mode_num))
                A = input_matrix()
                print("Input matrix:")
                pprint(A)
                A_transpose = A.T
                print("")
                print("A transpose: ")
                pprint(A_transpose)
                print("")
                save_matrix(A_transpose)
                print("")
                print("Result Saved!")
                print("")

            elif mode_num == 4:
                print("Mode {}: Determinant".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                try:
                    A_det = A.det()
                    print("Determinant of A:")
                    pprint(A_det)
                    print("")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => Det doesn't exit!")
                    print("")

            elif mode_num == 5:
                print("Mode {}: Row reduced Echelon Form".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                A_row_reduced = A.rref()[0]
                print("A row reduced:")
                pprint(A_row_reduced)
                print("")
                save_matrix(A_row_reduced)
                print("Result Saved!")
                print("")

            elif mode_num == 6:
                print("Mode {}: Basis for left and right nullspace".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                print("Basis for Right Nullspace: ")
                pprint(A.nullspace())
                print("")
                A_transpose = A.T
                print("Basis for Left Nullspace: ")
                pprint(A_transpose.nullspace())
                print("")
                print("Note: If these are empty, the nullspace is just")
                print("the trivial zero vector.")
                print("")


            elif mode_num == 7:
                print("Mode {}: Rank and basis for Row and Col space".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                print("Basis for column space: ")
                pprint(A.columnspace())
                print("")
                print("Basis for row space: ")
                pprint(A.rowspace())
                print("")
                print("Rank: {}".format(len(A.columnspace())))
                print("")


            elif mode_num == 8:
                print("Mode {}: Eigenvalues and Eigenvectors".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                try:
                    x = symbols('x')
                    char_pol = A.charpoly(x)
                    print("Characteristic Polynomial")
                    pprint(factor(char_pol.as_expr()))
                    print("")
                    pprint("Eigenvalues: {}".format(A.eigenvals()))
                    print("")
                    print("Warning: If the above is a really long output and the next part")
                    print("is taking a long time, then it's likely there are no rational")
                    print("non-zero solutions and you should just terminate the process.")
                    print("")
                    A_eigen_vects = A.eigenvects()
                    # pprint(A_eigen_vects)
                    # Save the eigen vectors
                    for eigen_basis in A_eigen_vects:
                        print("Eigenvalue: {}".format(eigen_basis[0]))
                        print("Eigenspace Basis: ")
                        pprint(eigen_basis[2])
                        for eigen_vect in eigen_basis[2]:
                            save_matrix(eigen_vect)

                    print("Eigen vectors saved!")
                    print("")

                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => No eigenvectors or values!")
                    print("")

            elif mode_num == 9:
                print("Mode {}: Diagonalize/Jordan Normal".format(mode_num))
                A = input_matrix()
                print("")
                print("Input matrix:")
                pprint(A)
                print("")
                try:
                    P, D = A.diagonalize()
                    print("Matrix is diagonalizable!")
                    print("PDP^-1 = A:")
                    print("P = ")
                    pprint(P)
                    save_matrix(P)
                    print("")
                    print("D = ")
                    pprint(D)
                    save_matrix(D)
                    print("")
                    print("Warning: if this hangs, then probably not easily diagonalizable.")
                    print("")


                except NonSquareMatrixError:
                    print("")
                    print("Error: Not a square matrix => Not diagonalizable!")
                    print("")

                except MatrixError:
                    print("Matrix is not diagonalizable!")
                    print("Trying Jordan Normal Form...")
                    print("")
                    P, J = A.jordan_form()
                    print("PJP^-1 = A:")
                    print("P = ")
                    pprint(P)
                    save_matrix(P)
                    print("")
                    print("J = ")
                    pprint(J)
                    save_matrix(J)
                    print("")
                    print("Warning: if this hangs, then probably not easily jordanable.")
                    print("")

            elif mode_num == 10:
                row_or_col_del_mode()

            elif mode_num == 11:
                applied_maths_mode()

            elif mode_num == 12:
                save_or_load_mode()

            elif mode_num == 420:
                sicko_mode()
                break

        except KeyboardInterrupt:
            print("")
            print("Returning to main menu...")
            continue



main()



