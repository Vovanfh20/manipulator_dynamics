# Geometry.py
import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

from Robot_parameters import dh_params, x_symb, y_symb, z_symb


class Link:
    """Звено манипулятора. Хранит все 4 DH-параметра."""

    def __init__(self, a, alpha, d, theta):
        self.a     = a
        self.alpha = alpha
        self.d     = d
        self.theta = theta

    def T_transform(self):
        """Матрица однородного преобразования 4x4 по параметрам Денавита–Хартенберга."""
        a, alpha, d, theta = self.a, self.alpha, self.d, self.theta
        T = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta),              0,            a            ],
            [sp.sin(theta)*sp.cos(alpha),  sp.cos(theta)*sp.cos(alpha), -sp.sin(alpha), -d*sp.sin(alpha)],
            [sp.sin(theta)*sp.sin(alpha),  sp.cos(theta)*sp.sin(alpha),  sp.cos(alpha),  d*sp.cos(alpha)],
            [0,                            0,                            0,              1               ]
        ])
        return T


class Manipulator_Geometry:
    """Геометрическая модель манипулятора на основе DH-параметров."""

    def __init__(self):
        self.links = []

    def add_link(self, a, alpha, d, theta):
        self.links.append(Link(a, alpha, d, theta))

    @classmethod
    def from_dh_params(cls, dh_params):
        robot = cls()
        for (a, alpha, d, theta) in dh_params:
            robot.add_link(a, alpha, d, theta)
        return robot

    def forward_geometry(self):
        """Итоговая матрица T = T1 * T2 * ... * Tn."""
        T_total = sp.eye(4)
        for link in self.links:
            T_total = T_total * link.T_transform()
        return sp.simplify(sp.trigsimp(T_total))


if __name__ == "__main__":
    robot = Manipulator_Geometry.from_dh_params(dh_params)

    T_final = robot.forward_geometry()

    x = sp.simplify(T_final[0, 3])
    y = sp.simplify(T_final[1, 3])
    z = sp.simplify(T_final[2, 3])

    print("Итоговая матрица T_final:")
    sp.pprint(T_final)

    display(sp.Eq(x_symb, x))
    display(sp.Eq(y_symb, y))
    display(sp.Eq(z_symb, z))
    
    # Формируем строки в LaTeX
    T_latex = sp.latex(T_final)
    x_latex = sp.latex(sp.Eq(x_symb, x))
    y_latex = sp.latex(sp.Eq(y_symb, y))
    z_latex = sp.latex(sp.Eq(z_symb, z))

    # Оборачиваем в \[ ... \] для display-режима LaTeX
    lines = [
        "% Итоговая матрица T_final:",
        r"\[" + T_latex + r"\]",
        "",
        "% Координаты:",
        r"\[" + x_latex + r"\]",
        r"\[" + y_latex + r"\]",
        r"\[" + z_latex + r"\]",
    ]

    output_path = "geometry_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Результаты записаны в {output_path}")
