from sympy import *
from sympy.matrices.common import MatrixError
from fractions import Fraction
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
        for i in range(len(saved_matrices)):
            print("")
            print("Saved Matrix: {}".format(i))
            pprint(saved_matrices[i])
            print("")
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
        

def get_input():
    """
    Takes both numerical and variable input 
    and returns the result as either a float or 
    a sympy symbol.
    Note: Also can take fractional input, e.g '1/2'.
    Note: For decimal input use fractions, if a float is given the
    integer part is used.
    """
    input_element = input("Element: ")
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
            print("New Matrix: ")
            R = int(input("No. Rows: ")) 
            C = int(input("No. Cols: "))

            try:
                # Initialize our matrix.
                A = Matrix([])
                for i in range(R):
                    ith_row = []
                    for j in range(C):
                        print("({}, {}) entry".format(i+1, j+1))
                        element = get_input()
                        ith_row.append(element)
                    A = A.row_insert(i, Matrix([ith_row]))

                save_matrix(A)
                return A

            except ValueError:
                print("Not a valid input!")
                # Returns None


def mode_selector():
    """
    Prints different available modes.
    Gets input.
    Checks for invalid input.
    """ 
    print("Available Operations:")
    print("[0]. Quit")
    print("[1]. Multiply Matrices")
    print("[2]. Inverse")
    print("[3]. Transpose")
    print("[4]. Determinant")
    print("[5]. Row reduce to echelon form.")
    print("[6]. Basis for left and right nullspace.")
    print("[7]. Rank and Basis for Row space and Column space")
    print("[8]. Eigenvalues and Eigenvectors")
    print("[9]. Diagonalize/Jordan Normal")
    print("[420]. Sicko mode")
    print("")
    available_nums = list(range(10))
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

def main():
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

Errors:
If an error occurs, or something takes too long to run,
just interupt the process and start again.
"""
    print(intro_text)

    while True:
        mode_num = mode_selector()

        if mode_num == 0:
            print("Bye!")
            break

        if mode_num == 1:
            A = input_matrix()
            print("Input matrix:")
            pprint(A)
            print("")
            print("Mode {} A * B:".format(mode_num))
            print("Please Input Matrix B")
            B = input_matrix() # get matrix B and save it
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
                print("Matrix Dimension Error!")
                # if ask_to_quit():
                #     break
                # else:
                #     pass
                # pass


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
                print("Determinant of A: {}".format(A_det))

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

        elif mode_num == 420:
            sicko_mode()
            break



main()

