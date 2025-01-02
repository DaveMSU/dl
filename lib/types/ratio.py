import typing as tp


class Ratio(float):
    def __new__(cls, value: tp.Any):
        value = float(value)
        if 0.0 <= value <= 1.0:
            return super().__new__(cls, float(value))
        else:
            raise ValueError("The value must be between 0 and 1 inclusive")

    def __add__(self, other):
        result = super().__add__(other)
        if 0 <= result <= 1:
            return Ratio(result)
        else:
            raise ValueError("The result is out of [0, 1] range")

    def __sub__(self, other):
        result = super().__sub__(other)
        if 0 <= result <= 1:
            return Ratio(result)
        else:
            raise ValueError("The result is out of [0, 1] range")

    def __mul__(self, other):
        result = super().__mul__(other)
        if 0 <= result <= 1:
            return Ratio(result)
        else:
            raise ValueError("The result is out of [0, 1] range")

    def __truediv__(self, other):
        result = super().__truediv__(other)
        if 0 <= result <= 1:
            return Ratio(result)
        else:
            raise ValueError("The result is out of [0, 1] range")

    def __repr__(self):
        return f"Ratio({super().__str__()})"
