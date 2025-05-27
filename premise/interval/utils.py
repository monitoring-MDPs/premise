from stormpy import Rational


def const(is_exact: bool, val: float):
    if is_exact:
        return Rational(val)
    else:
        return val
