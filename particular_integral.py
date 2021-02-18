# Using Sympy, plugs in a given particular integral
# into the homogenous part and returns the result

from sympy import *



x, y, l = symbols('x y l') # variables
while True:
    try:
        print("Enter Coefficients for Linear ODE of the form: ")
        print("A*y''' + B*y'' + C*y' + D*y")
        A = float(input('A: '))
        B = float(input('B: '))
        C = float(input('C: '))
        D = float(input('D: '))

        homo_solution = solve(A*(l**3) + B*(l**2) + C*l + D, l)
        print("Complimentary Equation Solutions: {}".format(homo_solution))

        a, b, c, d, e, f = symbols('a, b, c, d, e, f') # Constants for pi
        pi = sympify(input('Enter Particular Integral: '))
        LHS = A*diff((diff(diff(pi , x), x)), x) + B*diff((diff(pi, x)), x) + C*diff(pi, x) + D*pi
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("")
        print("LHS: {}".format(LHS))
        print("")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        exit_status = input("Exit? y/n: ")
        if exit_status == 'y':
            break
        elif exit_status == 'n':
            pass
        else:
            print("Not a valid answer, restarting...")
    except KeyboardInterrupt:
        break
