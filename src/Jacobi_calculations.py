# Jacobi_calculations.py
import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

from Robot_parameters import (
    dh_params,
    default_q, default_q_dot, det_J_symb,
    v_x_symb, v_y_symb, v_z_symb,
    x_symb, y_symb, z_symb,
)

from Robot_geometry import Manipulator_Geometry


class Manipulator_Jacobian:

    def __init__(self, dh_params):
        self.geometry = Manipulator_Geometry.from_dh_params(dh_params)

    def position_vector(self):
        """Вектор положения конца манипулятора из итоговой матрицы T."""
        T = self.geometry.forward_geometry()
        return sp.Matrix([T[0, 3], T[1, 3], T[2, 3]])

    def compute_jacobian(self, q_list):
        """Матрица Якоби J = dP/dq."""
        P = self.position_vector()
        J = sp.Matrix([
            [sp.diff(P[i], q_list[j]) for j in range(len(q_list))]
            for i in range(3)
        ])
        return sp.simplify(sp.trigsimp(J)), P

    def remove_zero_rows(self, J):
        """Удаляет строки, тождественно равные нулю."""
        kept_rows = [i for i in range(J.rows)
                     if any(J[i, j] != sp.S.Zero for j in range(J.cols))]
        return J[kept_rows, :], kept_rows

    def end_effector_velocity(self, J, q_dot_list):
        """Скорость конца манипулятора v = J * q_dot."""
        v = J * sp.Matrix(q_dot_list)
        return sp.simplify(sp.trigsimp(v))

    def generalized_velocities(self, J_red, kept_rows, v_symb_list):
        """Обобщённые скорости q_dot = J_red^-1 * v (только если J_red квадратная)."""
        rows, n = J_red.shape

        if rows != n:
            print("\n[!] Матрица Якоби не квадратная (%dx%d) — обращение невозможно." % (rows, n))
            return None, None

        v_red  = sp.Matrix([v_symb_list[i] for i in kept_rows])
        J_inv  = sp.simplify(sp.trigsimp(J_red.inv()))
        q_dot  = sp.simplify(sp.trigsimp(J_inv * v_red))
        return q_dot, J_inv

    def singularities(self, J_red, q_list):
        """Определитель и точки сингулярности."""
        rows, n = J_red.shape

        if rows != n:
            print("\n[!] Матрица Якоби не квадратная (%dx%d) — определитель не вычисляется." % (rows, n))
            return None, None

        det_expr     = sp.simplify(sp.trigsimp(J_red.det()))
        active_vars  = [q for q in q_list if q in det_expr.free_symbols]
        free_vars    = [q for q in q_list if q not in det_expr.free_symbols]

        if free_vars:
            print("\n[i] Переменные не входящие в det(J): %s" % free_vars)

        if not active_vars:
            print("\n[i] Определитель не зависит ни от одной координаты.")
            return det_expr, []

        sing_solutions = sp.solve(det_expr, active_vars)

        print("\nТочки сингулярности:")
        if sing_solutions:
            for sol in sing_solutions:
                if not isinstance(sol, (list, tuple)):
                    sol = (sol,)
                for var, val in zip(active_vars, sol):
                    display(sp.Eq(var, val))
        else:
            print("Аналитических решений не найдено.")

        return det_expr, sing_solutions


if __name__ == "__main__":

    robot = Manipulator_Jacobian(dh_params)

    # 1. Матрица Якоби
    print("\n" + "="*60)
    print("1. МАТРИЦА ЯКОБИ  J(q) = dP/dq")
    print("="*60)

    J, P = robot.compute_jacobian(default_q)

    print("\nВектор положения P(q):")
    display(sp.Eq(sp.Matrix([x_symb, y_symb, z_symb]), P))

    print("\nПолная матрица Якоби J(q):")
    sp.pprint(J)

    J_red, kept_rows = robot.remove_zero_rows(J)
    coord_names = ['x', 'y', 'z']
    removed = [coord_names[i] for i in range(3) if i not in kept_rows]

    if removed:
        print("\n[i] Исключены нулевые координаты: %s" % removed)
        print("\nУрезанная матрица Якоби J_red(q):")
        sp.pprint(J_red)

    # 2. Скорость конца манипулятора
    print("\n" + "="*60)
    print("2. СКОРОСТЬ КОНЦА МАНИПУЛЯТОРА  v = J(q) * q_dot")
    print("="*60)

    v_end = robot.end_effector_velocity(J, default_q_dot)

    display(sp.Eq(v_x_symb, v_end[0]))
    display(sp.Eq(v_y_symb, v_end[1]))
    display(sp.Eq(v_z_symb, v_end[2]))

    # 3. Обобщённые скорости
    print("\n" + "="*60)
    print("3. ОБОБЩЁННЫЕ СКОРОСТИ  q_dot = J_red^-1(q) * v")
    print("="*60)

    v_symb_list = [v_x_symb, v_y_symb, v_z_symb]
    q_dot_result, J_inv = robot.generalized_velocities(J_red, kept_rows, v_symb_list)

    if J_inv is not None:
        print("\nОбратная матрица J_red^-1(q):")
        sp.pprint(J_inv)

        print("\nОбобщённые скорости q_dot:")
        for q_d, res in zip(default_q_dot, q_dot_result):
            display(sp.Eq(q_d, res))

    # 4. Сингулярности
    print("\n" + "="*60)
    print("4. ТОЧКИ СИНГУЛЯРНОСТИ  det J_red(q) = 0")
    print("="*60)

    det_expr, sing_solutions = robot.singularities(J_red, default_q)

    if det_expr is not None:
        display(sp.Eq(det_J_symb, det_expr))
        display(sp.Eq(det_expr, 0))
