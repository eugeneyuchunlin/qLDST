import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


X, Y = sp.symbols("x y")
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


class Polynomial:
    """A polynomial over GF(2) in x and y.

    With orders=(l, m) it lives in the BB-code ring
    GF(2)[x, y] / (x^l - 1, y^m - 1): exponents are reduced mod (l, m),
    negative exponents are allowed and the transpose x -> x^-1, y -> y^-1
    is defined.

    Parsing and arithmetic are delegated to sympy (sympy.Poly with
    modulus=2). `terms` keeps the monomials in the order they were written,
    because syndrome-extraction schedules address A1, A2, A3 by position.
    Results of + and * come back in sympy's canonical order.
    """

    def __init__(
        self,
        polynomial_str: str = None,
        terms: list[tuple[int, int]] = None,
        orders: tuple[int, int] = None,
    ):
        if terms is not None and polynomial_str is not None:
            raise ValueError(
                "Polynomial accepts either polynomial_str or terms, not both."
            )
        if terms is None and polynomial_str is None:
            raise ValueError("Polynomial requires either polynomial_str or terms.")

        self.orders = tuple(orders) if orders is not None else None
        if terms is None:
            terms = self._parse_polynomial(polynomial_str)
        self.terms = self._normalize(terms)

    @staticmethod
    def _parse_polynomial(polynomial_str: str) -> list[tuple[int, int]]:
        # evaluate=False keeps the monomials in the order they were written.
        expr = parse_expr(
            polynomial_str, transformations=_TRANSFORMATIONS, evaluate=False
        )

        unknown = expr.free_symbols - {X, Y}
        if unknown:
            raise ValueError(f"Unknown symbols {unknown} in {polynomial_str!r}.")

        results = []
        for monomial in sp.Add.make_args(expr):
            coeff, rest = monomial.as_coeff_Mul()
            if not coeff.is_integer:
                raise ValueError(f"Non-integer coefficient in {monomial}.")
            powers = rest.as_powers_dict()
            exponent = (int(powers.get(X, 0)), int(powers.get(Y, 0)))
            # Coefficients are taken mod 2; _normalize cancels repeats.
            results.extend([exponent] * (int(coeff) % 2))

        return results

    def _normalize(self, terms) -> list[tuple[int, int]]:
        """Reduce exponents into the ring and cancel repeated monomials mod 2,
        keeping first-occurrence order."""
        parity = {}
        for exp_x, exp_y in terms:
            exp_x, exp_y = int(exp_x), int(exp_y)
            if self.orders is not None:
                exp_x %= self.orders[0]
                exp_y %= self.orders[1]
            elif exp_x < 0 or exp_y < 0:
                raise ValueError(
                    "Negative exponents require orders=(l, m) to be set."
                )
            parity[(exp_x, exp_y)] = parity.get((exp_x, exp_y), 0) ^ 1
        return [term for term, bit in parity.items() if bit]

    # ----- sympy interop -----

    @property
    def poly(self) -> sp.Poly:
        return sp.Poly.from_dict(
            {term: 1 for term in self.terms}, X, Y, modulus=2
        )

    def as_expr(self) -> sp.Expr:
        """Plain sympy expression, e.g. for qldpc.codes.BBCode."""
        return self.poly.as_expr()

    @classmethod
    def from_poly(cls, poly: sp.Poly, orders: tuple[int, int] = None):
        # The zero Poly reports monoms() == [(0, 0)], so read nonzero terms.
        terms = [monom for monom, coeff in poly.terms() if coeff % 2]
        return cls(terms=terms, orders=orders)

    def with_orders(self, orders: tuple[int, int]) -> "Polynomial":
        """The same polynomial viewed in GF(2)[x, y] / (x^l - 1, y^m - 1)."""
        return Polynomial(terms=self.terms, orders=orders)

    # ----- algebra -----

    def _common_orders(self, other: "Polynomial"):
        if self.orders is None:
            return other.orders
        if other.orders is not None and other.orders != self.orders:
            raise ValueError(
                f"Mismatched orders {self.orders} and {other.orders}."
            )
        return self.orders

    def __add__(self, other: "Polynomial") -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        return Polynomial.from_poly(
            self.poly + other.poly, self._common_orders(other)
        )

    __sub__ = __add__  # characteristic 2

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        return Polynomial.from_poly(
            self.poly * other.poly, self._common_orders(other)
        )

    def transpose(self) -> "Polynomial":
        """x -> x^-1, y -> y^-1, i.e. the matrix transpose of the lift.
        Term order is preserved."""
        if self.orders is None:
            raise ValueError("transpose requires orders=(l, m) to be set.")
        return Polynomial(
            terms=[(-exp_x, -exp_y) for exp_x, exp_y in self.terms],
            orders=self.orders,
        )

    @property
    def T(self) -> "Polynomial":
        return self.transpose()

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.orders == other.orders and set(self.terms) == set(other.terms)

    def __bool__(self):
        return bool(self.terms)

    def __str__(self):
        return str(self.as_expr())

    def __repr__(self):
        return f"Polynomial(terms={self.terms}, orders={self.orders})"


# class LogicalOperators:
# 
#     alpha_string = ["1", "x^3y^5", "x^11"]