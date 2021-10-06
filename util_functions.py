from Polynomial import Polynomial
import numpy as np


def get_inverse(c, field_value=257):
    inv = None
    for i in range(1, field_value):
        if (c * i) % field_value == 1:
            inv = i
            break
    if inv is None:
        raise ValueError("The value of inv is None")
    return inv


def get_Lagrange_Polynomials(e, field_value=257):
    k = len(e)
    Lagrange_polynomials = []
    for i in range(k):
        temp_prod = 1
        temp_poly = Polynomial([1])
        for j in range(k):
            if (i != j):
                temp_poly = temp_poly.multiply(Polynomial([1, (field_value - e[j]) % field_value]))
                temp_val = ((e[i] - e[j] + field_value) % field_value)
                inverse_temp_val = get_inverse(temp_val)
                temp_poly = temp_poly.multiply(Polynomial([inverse_temp_val]))
        Lagrange_polynomials += [temp_poly]
    return Lagrange_polynomials


def get_prod_funs(e, field_value=257):
    k = len(e)
    temp_poly = Polynomial([1])
    for i in range(k):
        temp_poly = temp_poly.multiply(Polynomial([1, (field_value - e[i]) % field_value]))
    return temp_poly


def reconstruct_polynomial(alpha, poly_alpha, field_value=257):
    temp_poly = Polynomial([0])
    t = len(alpha)
    for i in range(t):
        poly = Polynomial([1])
        temp_prod = 1
        alpha_i = alpha[i]
        for j in range(t):
            if (i != j):
                poly = poly.multiply(Polynomial([1, (field_value - alpha[j]) % field_value]))
                temp_prod = temp_prod * ((alpha_i - alpha[j] + field_value) % field_value)
                temp_prod %= field_value
        poly = poly.divide_by_constant(temp_prod)
        poly = poly.multiply(Polynomial([poly_alpha[i]]))
        temp_poly = temp_poly.add(poly)
    return temp_poly
