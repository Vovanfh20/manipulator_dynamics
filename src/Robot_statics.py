import sympy as sp
from IPython.display import display
sp.init_printing(use_latex=True)

from Robot_parameters import dh_params, joint_types, f_x, f_y, f_z, n_x, n_y, n_z 
from Robot_velocities import Link

class Robot_static:
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

    def static_calculation(self):
        n = len(self.links)
        f = [sp.Matrix([0, 0, 0]) for _ in range(n)]
        n_vec = [sp.Matrix([0, 0, 0]) for _ in range(n)]

        # Силы и моменты на схвате (в последней системе координат)
        f[n-1] = sp.Matrix([f_x, f_y, f_z])
        n_vec[n-1] = sp.Matrix([0, 0, 0])

        # Обратный цикл от схвата к базе
        for i in range(n - 1, 0, -1):
            R = self.links[i].R_rotation()
            P = self.links[i].P_coordinates()
            f[i - 1] = R * f[i]
            n_vec[i - 1] = R * n_vec[i] + P.cross(f[i - 1])
        
        # Прямой цикл для расчёта tau, сколько обобщённых координат, столько и tau
        n_joints = len(joint_types) 
        z = sp.Matrix([0, 0, 1])
        tau = []
        for i in range(n_joints):
            if joint_types[i] == 'R':
                tau.append(sp.simplify(z.dot(n_vec[i])))
            else:
                tau.append(sp.simplify(z.dot(f[i])))
        return tau, f, n_vec    # возвращаем всё


if __name__ == "__main__":
    robot = Robot_static.from_dh_params(dh_params)
    torques, forces, moments = robot.static_calculation()

    print("Обобщённые силы в суставах:")
    for i, tau in enumerate(torques, start=1):
        display(sp.Eq(sp.Symbol(f'tau_{i}'), tau))

    print("\nСилы и моменты в системах координат (от схвата к базе):")
    for i in range(len(forces), 0, -1):
        print(f"\n--- Система {i} ---")
        
        # Момент n_i (компоненты x, y, z)
        nx, ny, nz = sp.symbols(f'n_{i}x n_{i}y n_{i}z')
        display(sp.Eq(nx, sp.simplify(moments[i - 1][0])))
        display(sp.Eq(ny, sp.simplify(moments[i - 1 ][1])))
        display(sp.Eq(nz, sp.simplify(moments[i - 1][2])))
        
        # Сила f_i (компоненты x, y, z)
        fx, fy, fz = sp.symbols(f'f_{i}x f_{i}y f_{i}z')
        display(sp.Eq(fx, sp.simplify(forces[i - 1][0])))
        display(sp.Eq(fy, sp.simplify(forces[i - 1][1])))
        display(sp.Eq(fz, sp.simplify(forces[i - 1][2])))
    
    # ── Экспорт в LaTeX ────────────────────────────────────────────────────────
    lines = []

    lines.append(r"\section*{Обобщённые силы в суставах}")
    for i, tau in enumerate(torques, start=1):
        tau_symb = sp.Symbol(f'tau_{i}')
        lines.append(r"\[" + sp.latex(sp.Eq(tau_symb, tau)) + r"\]")
    lines.append("")

    lines.append(r"\section*{Силы и моменты в системах координат (от схвата к базе)}")
    for i in range(len(forces), 0, -1):
        lines.append(rf"\subsection*{{Система {i}}}")

        nx, ny, nz = sp.symbols(f'n_{i}x n_{i}y n_{i}z')
        lines.append(r"\textbf{Момент:}")
        lines.append(r"\[" + sp.latex(sp.Eq(nx, sp.simplify(moments[i - 1][0]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(ny, sp.simplify(moments[i - 1][1]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(nz, sp.simplify(moments[i - 1][2]))) + r"\]")

        fx, fy, fz = sp.symbols(f'f_{i}x f_{i}y f_{i}z')
        lines.append(r"\textbf{Сила:}")
        lines.append(r"\[" + sp.latex(sp.Eq(fx, sp.simplify(forces[i - 1][0]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(fy, sp.simplify(forces[i - 1][1]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(fz, sp.simplify(forces[i - 1][2]))) + r"\]")
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

    with open("statics_output.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)

    print("Результаты записаны в statics_output.tex")
             
        
        