# kinematics_robot.py
import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

from Robot_parameters import (
    dh_params,
    v_x_symb, v_y_symb, v_z_symb,
    default_theta_point, default_d_point
)


class Link:
    """Звено манипулятора. Хранит все 4 DH-параметра."""

    def __init__(self, a, alpha, d, theta):
        self.a     = a
        self.alpha = alpha
        self.d     = d
        self.theta = theta

    def R_rotation(self):
        """Матрица поворота 3x3 для данного звена."""
        theta, alfa = self.theta, self.alpha
        R = sp.Matrix([
            [sp.cos(theta),                -sp.sin(theta),               0           ],
            [sp.sin(theta)*sp.cos(alfa),     sp.cos(theta)*sp.cos(alfa),    -sp.sin(alfa)  ],
            [sp.sin(theta)*sp.sin(alfa),     sp.cos(theta)*sp.sin(alfa),     sp.cos(alfa)  ],
        ])
        return R

    def P_coordinates(self):
        """Вектор положения начала следующей СК относительно текущей."""
        return sp.Matrix([
            self.a,
            -self.d * sp.sin(self.alpha),
             self.d * sp.cos(self.alpha)
        ])

    def Theta_point_Z(self, theta_point):
        """Вклад вращательного сочленения в угловую скорость (вдоль оси Z)."""
        return sp.Matrix([0, 0, theta_point])

    def D_point_Z(self, d_point):
        """Вклад поступательного сочленения в линейную скорость (вдоль оси Z)."""
        return sp.Matrix([0, 0, d_point])


class Manipulator_kinematics:
    """Кинематическая модель манипулятора (итерационный метод Крейга)."""

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

    def forward_kinematics(self, theta_point_list, d_point_list):
        """
        Итерационный расчёт скоростей всех звеньев.
        theta_point_list — угловые скорости сочленений
        d_point_list     — линейные скорости сочленений

        Возвращает:
            omega_list    — [omega_0, omega_1, ..., omega_n]
                            индекс 0 = основание (нули), индекс i = система {i}
            v_list — [v_0, v_1, ..., v_n]
                            индекс 0 = основание (нули), индекс i = система {i}
        """
        n = len(self.links)

        omega_list    = [sp.Matrix([0, 0, 0]) for _ in range(n + 1)]
        v_list = [sp.Matrix([0, 0, 0]) for _ in range(n + 1)]
        
        omega_list[0] = sp.Matrix([0, 0, 0])
        v_list[0] = sp.Matrix([0, 0, 0])
        
        for i, link in enumerate(self.links):
            R_T = link.R_rotation().T
            P   = link.P_coordinates()

            omega_list[i+1]    = R_T * omega_list[i] + link.Theta_point_Z(theta_point_list[i])
            v_list[i+1] = R_T * (v_list[i] + omega_list[i].cross(P)) + link.D_point_Z(d_point_list[i])

        return omega_list, v_list

    def Rotation_total(self):
        """Итоговая матрица поворота R = R1 * R2 * ... * Rn."""
        R_total = sp.eye(3)
        for link in self.links:
            R_total = R_total * link.R_rotation()
        return R_total


if __name__ == "__main__":
    robot = Manipulator_kinematics.from_dh_params(dh_params)
    n = len(robot.links)
    R_final = robot.Rotation_total()
    print("Итоговая матрица поворота R_final:")
    sp.pprint(R_final)

    omega_list, v_list = robot.forward_kinematics(default_theta_point, default_d_point)
    
    # Вывод матриц P
    print("\nВекторы положения P (от i-1 до i в системе {i-1}):")
    for i, link in enumerate(robot.links, start=1):
        P = link.P_coordinates()
        px, py, pz = sp.symbols(f'P_{i}x P_{i}y P_{i}z')
        display(sp.Eq(px, sp.simplify(P[0])))
        display(sp.Eq(py, sp.simplify(P[1])))
        display(sp.Eq(pz, sp.simplify(P[2])))     
    
    # Скорость конца манипулятора в базовой системе координат
    v_end_base = sp.simplify(sp.trigsimp(robot.Rotation_total() * v_list[n]))

    display(sp.Eq(v_x_symb, v_end_base[0]))
    display(sp.Eq(v_y_symb, v_end_base[1]))
    display(sp.Eq(v_z_symb, v_end_base[2]))

    # Вывод скоростей каждого звена покомпонентно (LaTeX-стиль) 
    for i in range(1, len(omega_list)):
        om = sp.simplify(sp.trigsimp(omega_list[i]))
        v  = sp.simplify(sp.trigsimp(v_list[i]))

        ox, oy, oz = sp.symbols(f'omega_{i}x omega_{i}y omega_{i}z')
        vx, vy, vz = sp.symbols(f'v_{i}x v_{i}y v_{i}z')

        display(sp.Eq(ox, om[0])); display(sp.Eq(oy, om[1])); display(sp.Eq(oz, om[2]))
        display(sp.Eq(vx, v[0]));  display(sp.Eq(vy, v[1]));  display(sp.Eq(vz, v[2])) 
   
     
   
    # ── Экспорт в LaTeX ────────────────────────────────────────────────────────
    lines = []

    lines.append(r"\section*{Итоговая матрица поворота $R_{final}$}")
    lines.append(r"\[" + sp.latex(R_final) + r"\]")
    lines.append("")

    lines.append(r"\section*{Скорость конца манипулятора (базовая СК)}")
    lines.append(r"\[" + sp.latex(sp.Eq(v_x_symb, v_end_base[0])) + r"\]")
    lines.append(r"\[" + sp.latex(sp.Eq(v_y_symb, v_end_base[1])) + r"\]")
    lines.append(r"\[" + sp.latex(sp.Eq(v_z_symb, v_end_base[2])) + r"\]")
    lines.append("")

    lines.append(r"\section*{Угловые и линейные скорости звеньев}")
    for i in range(1, len(omega_list)):
        om = sp.simplify(sp.trigsimp(omega_list[i]))
        v  = sp.simplify(sp.trigsimp(v_list[i]))
        ox, oy, oz = sp.symbols(f'omega_{i}x omega_{i}y omega_{i}z')
        vx, vy, vz = sp.symbols(f'v_{i}x v_{i}y v_{i}z')

        lines.append(rf"\subsection*{{Звено {i}}}")
        lines.append(r"\[" + sp.latex(sp.Eq(ox, om[0])) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(oy, om[1])) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(oz, om[2])) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(vx, v[0])) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(vy, v[1])) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(vz, v[2])) + r"\]")
        lines.append("")
    
    lines.append(r"\section*{Векторы положения $P_i$}")
    for i, link in enumerate(robot.links, start=1):
        P = link.P_coordinates()
        px, py, pz = sp.symbols(f'P_{i}x P_{i}y P_{i}z')
        lines.append(rf"\subsection*{{Звено {i}}}")
        lines.append(r"\[" + sp.latex(sp.Eq(px, sp.simplify(P[0]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(py, sp.simplify(P[1]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(pz, sp.simplify(P[2]))) + r"\]")
        lines.append("")
    
    tex_content = (
        r"\documentclass{article}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[russian]{babel}" + "\n"
        r"\usepackage{amsmath}" + "\n"
        r"\begin{document}" + "\n\n"
        + "\n".join(lines) + "\n\n"
        r"\end{document}"
    )

    with open("kinematics_output.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)

    print("Результаты записаны в kinematics_output.tex")