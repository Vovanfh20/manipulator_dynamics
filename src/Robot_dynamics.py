# Robot_dynamics.py
import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

from Robot_parameters import (
    dh_params,
    joint_types,
    masses,
    inertia_tensors,
    P_Center_coordinates,
    q_symbols, dot_theta1, dot_theta2, dot_d3, ddot_theta1, ddot_theta2, ddot_d3, dot_d2
)
from Robot_accelerations import Link   # Link из файла ускорений


class Robot_dynamics:
    """
    Расчёт обобщённых сил методом Ньютона–Эйлера по Крейгу.
    Алгоритм:
      1) по кинематике/ускорениям получаем F_i и N_i для каждого звена;
      2) затем идём от схвата к базе и находим f_i, n_i и tau_i;
      3) разложение tau = M(q)*q̈ + C(q,q̇)*q̇ + G(q).
    """

    def __init__(self):
        self.links = []

    def add_link(self, a, alpha, d, theta, p_center):
        self.links.append(Link(a, alpha, d, theta, p_center))

    @classmethod
    def from_dh_params(cls, dh_params, p_center_list):
        robot = cls()
        p_centers = list(p_center_list)
        while len(p_centers) < len(dh_params):
            p_centers.append((0, 0, 0))
        for (a, alpha, d, theta), p_center in zip(dh_params, p_centers):
            robot.add_link(a, alpha, d, theta, p_center)
        return robot

    @staticmethod
    def _pad_list(seq, target_len, pad_value):
        out = list(seq)
        while len(out) < target_len:
            out.append(pad_value)
        return out

    def _compute_inertial_forces_and_torques(self, omega_list, epsilon_list, a_c_list):
        n = len(self.links)
        mass_list = self._pad_list(masses, n, 0)
        inertia_list = self._pad_list(inertia_tensors, n, sp.zeros(3))

        F_list = [None]
        N_list = [None]

        for i in range(1, n + 1):
            m_i = mass_list[i - 1]
            I_i = inertia_list[i - 1]
            omega_i   = omega_list[i]
            epsilon_i = epsilon_list[i]
            a_c_i     = a_c_list[i]

            F_i = m_i * a_c_i
            N_i = I_i * epsilon_i + omega_i.cross(I_i * omega_i)

            F_list.append(sp.simplify(F_i))
            N_list.append(sp.simplify(N_i))

        return F_list, N_list

    def _inward_iteration(self, F_list, N_list, external_wrench=None):
        n_links  = len(self.links)
        n_joints = len(joint_types)

        f     = [sp.Matrix([0, 0, 0]) for _ in range(n_links + 2)]
        n_vec = [sp.Matrix([0, 0, 0]) for _ in range(n_links + 2)]

        if external_wrench is not None:
            f[n_links + 1], n_vec[n_links + 1] = external_wrench
        else:
            f[n_links + 1]     = sp.Matrix([0, 0, 0])
            n_vec[n_links + 1] = sp.Matrix([0, 0, 0])

        tau = [sp.Integer(0)] * n_joints
        z   = sp.Matrix([0, 0, 1])

        for i in range(n_links, 0, -1):
            if i < n_links:
                R_next_T = self.links[i].R_rotation()
                P_i_plus = self.links[i].P_coordinates()
            else:
                R_next_T = sp.eye(3)
                P_i_plus = sp.Matrix([0, 0, 0])

            P_C_i = self.links[i - 1].P_center()
            F_i   = F_list[i]
            N_i   = N_list[i]

            f_i = F_i + R_next_T * f[i + 1]
            n_i = (
                N_i
                + R_next_T * n_vec[i + 1]
                + P_C_i.cross(F_i)
                + P_i_plus.cross(R_next_T * f[i + 1])
            )

            f[i]     = sp.simplify(f_i)
            n_vec[i] = sp.simplify(n_i)

            if i <= n_joints:
                jt = joint_types[i - 1]
                if jt == 'R':
                    tau[i - 1] = sp.simplify(n_i.dot(z))
                else:
                    tau[i - 1] = sp.simplify(f_i.dot(z))

        return tau, f, n_vec

    def compute_dynamics(self, omega_list, epsilon_list, a_c_list, external_wrench=None):
        F_list, N_list = self._compute_inertial_forces_and_torques(
            omega_list, epsilon_list, a_c_list
        )
        tau, f, n_vec = self._inward_iteration(
            F_list, N_list, external_wrench=external_wrench
        )
        tau = [sp.trigsimp(sp.simplify(t)) for t in tau]
        return tau, f, n_vec

    # ══════════════════════════════════════════════════════════════════════════
    # Извлечение матриц M, C, G
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_MCG(tau, q_ddot_list, q_dot_list, q_list=None):
        """
        Разлагает вектор обобщённых сил τ в форму
            τ = M(q)·q̈  +  C(q,q̇)·q̇  +  G(q)

        Параметры
        ----------
        tau        : list[Expr]  — выходные обобщённые силы (длина n)
        q_ddot_list: list[Symbol] — обобщённые ускорения  q̈₁…q̈ₙ
        q_dot_list : list[Symbol] — обобщённые скорости   q̇₁…q̇ₙ
        q_list     : list[Symbol] или None
                     Если переданы — вычисляется матрица C через символы
                     Кристоффеля (точное разложение).
                     Если None — возвращается только вектор h = C·q̇.

        Возвращает
        ----------
        M     : Matrix(n×n)  — матрица масс/инерций
        C_mat : Matrix(n×n) или None
        h_vec : Matrix(n×1)  — вектор C(q,q̇)·q̇  (всегда доступен)
        G_vec : Matrix(n×1)  — вектор гравитационных сил
        """
        n       = len(tau)
        q_ddot  = list(q_ddot_list)
        q_dot   = list(q_dot_list)

        print("  [MCG] Вычисляю матрицу M ...")
        # ── 1. Матрица масс M ─────────────────────────────────────────────
        # τ линейно по q̈  →  M[i,j] = ∂τ_i / ∂q̈_j
        M = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = sp.diff(tau[i], q_ddot[j])
        M = sp.simplify(M)

        print("  [MCG] Вычисляю вектор G ...")
        # ── 2. Вектор гравитации G ────────────────────────────────────────
        # G_i = τ_i при q̇ = 0, q̈ = 0
        subs_zero = {s: 0 for s in q_dot + q_ddot}
        G_vec = sp.Matrix([
            sp.trigsimp(sp.simplify(t.subs(subs_zero)))
            for t in tau
        ])

        print("  [MCG] Вычисляю вектор h = C·q̇ ...")
        # ── 3. Вектор кориолиса/центробежных h = C(q,q̇)·q̇ ──────────────
        # Убираем q̈, вычитаем G
        subs_no_ddot = {s: 0 for s in q_ddot}
        tau_no_ddot  = [
            sp.trigsimp(sp.simplify(t.subs(subs_no_ddot)))
            for t in tau
        ]
        h_vec = sp.Matrix([
            sp.trigsimp(sp.simplify(tau_no_ddot[i] - G_vec[i]))
            for i in range(n)
        ])

        # ── 4. Матрица C через символы Кристоффеля ────────────────────────
        # C[i,j] = Σ_k  Γ_{kij}(q) · q̇_k
        # Γ_{kij} = ½ (∂M_{ij}/∂q_k  +  ∂M_{ik}/∂q_j  −  ∂M_{jk}/∂q_i)
        C_mat = None
        if q_list is not None:
            print("  [MCG] Вычисляю матрицу C (символы Кристоффеля) ...")
            q     = list(q_list)
            C_mat = sp.zeros(n, n)
            for i in range(n):
                for j in range(n):
                    c_ij = sp.Integer(0)
                    for k in range(n):
                        gamma_kij = sp.Rational(1, 2) * (
                            sp.diff(M[i, j], q[k])
                            + sp.diff(M[i, k], q[j])
                            - sp.diff(M[j, k], q[i])
                        )
                        c_ij += gamma_kij * q_dot[k]
                    C_mat[i, j] = sp.trigsimp(sp.simplify(c_ij))

        print("  [MCG] Готово.")
        return M, C_mat, h_vec, G_vec


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from Robot_velocities import Manipulator_kinematics
    from Robot_accelerations import Manipulator_Acceleration
    from Robot_parameters import (
        default_theta_point, default_d_point,
        default_theta_ddot,  default_d_ddot,
        # ──────────────────────────────────────────────────────────────────
        # Добавьте в Robot_parameters.py список обобщённых координат,
        # например:  q_symbols = [theta_1, theta_2, ...]
        # и раскомментируйте строку ниже для построения матрицы C.
        # q_symbols,
        # ──────────────────────────────────────────────────────────────────
    )

    # ── Формируем списки символов ─────────────────────────────────────────
    # q̇  — объединяем theta_point и d_point в порядке суставов
    # ── Формируем списки символов ─────────────────────────────────────────
    q_dot_syms  = [dot_theta1, dot_theta2, dot_d3]
    # Для q'' аналогично
    q_ddot_syms = [ddot_theta1, ddot_theta2, ddot_d3]
    n_j = len(joint_types)

   
    n_j          = len(joint_types)
    q_dot_syms   = q_dot_syms[:n_j]
    q_ddot_syms  = q_ddot_syms[:n_j]

    # ── 1. Кинематика ─────────────────────────────────────────────────────
    print("Прямая кинематика...")
    kin = Manipulator_kinematics.from_dh_params(dh_params)
    omega_list, v_list = kin.forward_kinematics(default_theta_point, default_d_point)

    # ── 2. Ускорения ──────────────────────────────────────────────────────
    print("Ускорения...")
    acc = Manipulator_Acceleration.from_dh_params(dh_params, P_Center_coordinates)
    epsilon_list, a_list, a_c_list = acc.forward_acceleration(
        omega_list=omega_list,
        ddot_theta_list=default_theta_ddot,
        ddot_d_list=default_d_ddot,
        dot_theta_list=default_theta_point,
        dot_d_list=default_d_point,
    )

    # ── 3. Динамика (обратный ход) ────────────────────────────────────────
    print("Динамика (Ньютон–Эйлер)...")
    dyn = Robot_dynamics.from_dh_params(dh_params, P_Center_coordinates)
    tau, forces, moments = dyn.compute_dynamics(omega_list, epsilon_list, a_c_list)
    
    external_wrench = (sp.Matrix([0, 0, 0]), sp.Matrix([0, 0, 0]))
    tau, forces, moments = dyn.compute_dynamics(
        omega_list, epsilon_list, a_c_list,
        external_wrench=external_wrench)
    
    # ── 4. Извлечение M, C, G ─────────────────────────────────────────────
    print("Разложение τ = M·q̈ + C·q̇ + G ...")
    M_mat, C_mat, h_vec, G_vec = Robot_dynamics.extract_MCG(
        tau,
        q_ddot_syms,
        q_dot_syms,
        q_list=q_symbols,   
    )

    # ── 5. Вывод на экран (красивый символьный вид) ───────────────────────
    q_ddot_vec = sp.Matrix(q_ddot_syms)
    q_dot_vec  = sp.Matrix(q_dot_syms)
    tau_reconstructed = M_mat * q_ddot_vec + h_vec + G_vec

    def _display_matrix_elements(mat, name, rows=None, cols=None):
        """
        Выводит матрицу целиком (display), затем каждый элемент отдельной строкой:
            M_{11} = ...,  M_{12} = ...  и т.д.
        rows/cols — метки строк и столбцов (по умолчанию 1..n).
        """
        r, c = mat.shape
        row_idx = rows if rows else [str(i + 1) for i in range(r)]
        col_idx = cols if cols else [str(j + 1) for j in range(c)]

        # Матрица целиком
        lhs_mat = sp.MatrixSymbol(name, r, c)
        display(sp.Eq(lhs_mat, mat, evaluate=False))

        # Каждый элемент отдельно
        for i in range(r):
            for j in range(c):
                elem = mat[i, j]
                if elem == 0:          # не засорять вывод нулями
                    continue
                lhs = sp.Symbol(f'{name}_{{{row_idx[i]}{col_idx[j]}}}')
                display(sp.Eq(lhs, elem))

    def _display_vector_elements(vec, name):
        """
        Выводит вектор целиком, затем каждый элемент: G_1 = ..., G_2 = ...
        """
        n = vec.shape[0]
        lhs_vec = sp.MatrixSymbol(name, n, 1)
        display(sp.Eq(lhs_vec, vec, evaluate=False))
        for i in range(n):
            elem = vec[i]
            if elem == 0:
                continue
            lhs = sp.Symbol(f'{name}_{{{i + 1}}}')
            display(sp.Eq(lhs, elem))

    # ── 5a. Общая форма ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Уравнение динамики:  τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)")
    print("═" * 60)
    tau_lhs = sp.MatrixSymbol(r'\tau', n_j, 1)
    M_sym   = sp.MatrixSymbol('M',    n_j, n_j)
    C_sym   = sp.MatrixSymbol('C',    n_j, n_j)
    G_sym   = sp.MatrixSymbol('G',    n_j,  1)
    display(sp.Eq(tau_lhs,
                  M_sym * sp.MatrixSymbol(r'\ddot{q}', n_j, 1)
                  + C_sym * sp.MatrixSymbol(r'\dot{q}', n_j, 1)
                  + G_sym,
                  evaluate=False))

    # ── 5b. Матрица масс M ────────────────────────────────────────────────
    print("\n── Матрица масс  M(q)  [размер {}×{}] ──".format(n_j, n_j))
    _display_matrix_elements(M_mat, 'M')

    # ── 5c. Вектор / матрица кориолиса ────────────────────────────────────
    print("\n── Вектор кориолиса/центробежных  h = C·q̇  [размер {}×1] ──".format(n_j))
    _display_vector_elements(h_vec, 'h')

    if C_mat is not None:
        print("\n── Матрица кориолиса  C(q,q̇)  [размер {}×{}] ──".format(n_j, n_j))
        _display_matrix_elements(C_mat, 'C')

    # ── 5d. Вектор гравитации G ───────────────────────────────────────────
    print("\n── Вектор гравитации  G(q)  [размер {}×1] ──".format(n_j))
    _display_vector_elements(G_vec, 'G')

    # ── 5e. Поэлементное уравнение τ_i = ... ──────────────────────────────
    print("\n── Поэлементное уравнение динамики ──")
    for i in range(n_j):
        lhs  = sp.Symbol(r'\tau_{' + str(i + 1) + '}')
        expr = sp.trigsimp(sp.simplify(tau_reconstructed[i]))
        display(sp.Eq(lhs, expr))

    # ── 6. Экспорт в LaTeX ────────────────────────────────────────────────
    lines = []

    # ── вспомогательные функции для LaTeX ─────────────────────────────────
    def _latex_matrix_full(mat, name):
        """Матрица целиком в окружении pmatrix."""
        return r"\[" + name + r" = " + sp.latex(mat, mat_str='pmatrix') + r"\]"

    def _latex_matrix_elements(mat, name, rows=None, cols=None):
        """Каждый ненулевой элемент отдельной строкой."""
        r, c = mat.shape
        row_idx = rows if rows else [str(i + 1) for i in range(r)]
        col_idx = cols if cols else [str(j + 1) for j in range(c)]
        out = []
        for i in range(r):
            for j in range(c):
                elem = mat[i, j]
                if elem == 0:
                    continue
                lhs_str = f'{name}_{{{row_idx[i]}{col_idx[j]}}}'
                lhs_sym = sp.Symbol(lhs_str)
                out.append(r"\[" + sp.latex(sp.Eq(lhs_sym, elem)) + r"\]")
        return out

    def _latex_vector_elements(vec, name):
        """Каждый ненулевой элемент вектора отдельной строкой."""
        out = []
        for i in range(vec.shape[0]):
            elem = vec[i]
            if elem == 0:
                continue
            lhs_sym = sp.Symbol(f'{name}_{{{i + 1}}}')
            out.append(r"\[" + sp.latex(sp.Eq(lhs_sym, elem)) + r"\]")
        return out

    # 6a. Общая форма
    lines.append(r"\section*{Уравнение динамики манипулятора}")
    lines.append(
        r"\[\boldsymbol{\tau} = M(\mathbf{q})\,\ddot{\mathbf{q}}"
        r" + C(\mathbf{q},\dot{\mathbf{q}})\,\dot{\mathbf{q}}"
        r" + G(\mathbf{q})\]"
    )
    lines.append(r"где $n = " + str(n_j) + r"$ степеней свободы.")
    lines.append("")

    # 6b. Матрица масс M — целиком, затем поэлементно
    lines.append(r"\section*{Матрица масс $M(q)$ — размер $"
                 + str(n_j) + r"\times " + str(n_j) + r"$}")
    lines.append(_latex_matrix_full(M_mat, 'M'))
    lines.append(r"\subsection*{Элементы матрицы $M$}")
    lines.extend(_latex_matrix_elements(M_mat, 'M'))
    lines.append("")

    # 6c. Вектор h = C·q̇ — целиком, затем поэлементно
    lines.append(r"\section*{Вектор кориолиса/центробежных $h = C(q,\dot{q})\dot{q}$"
                 r" — размер $" + str(n_j) + r"\times 1$}")
    lines.append(_latex_matrix_full(h_vec, 'h'))
    lines.append(r"\subsection*{Элементы вектора $h$}")
    lines.extend(_latex_vector_elements(h_vec, 'h'))
    lines.append("")

    # 6d. Матрица C (если вычислена)
    if C_mat is not None:
        lines.append(r"\section*{Матрица Кориолиса $C(q,\dot{q})$"
                     r" — размер $" + str(n_j) + r"\times " + str(n_j) + r"$}")
        lines.append(_latex_matrix_full(C_mat, 'C'))
        lines.append(r"\subsection*{Элементы матрицы $C$}")
        lines.extend(_latex_matrix_elements(C_mat, 'C'))
        lines.append("")

    # 6e. Вектор гравитации G — целиком, затем поэлементно
    lines.append(r"\section*{Вектор гравитации $G(q)$"
                 r" — размер $" + str(n_j) + r"\times 1$}")
    lines.append(_latex_matrix_full(G_vec, 'G'))
    lines.append(r"\subsection*{Элементы вектора $G$}")
    lines.extend(_latex_vector_elements(G_vec, 'G'))
    lines.append("")

    # 6f. Поэлементное уравнение τ_i = (M·q̈)_i + h_i + G_i
    lines.append(r"\section*{Поэлементное уравнение динамики}")
    for i in range(n_j):
        lhs  = sp.Symbol(r'\tau_{' + str(i + 1) + '}')
        expr = sp.trigsimp(sp.simplify(tau_reconstructed[i]))
        lines.append(r"\[" + sp.latex(sp.Eq(lhs, expr)) + r"\]")
    lines.append("")

    # 6g. Старые результаты: силы и моменты
    lines.append(r"\section*{Силы и моменты (от схвата к базе)}")
    for i in range(len(forces) - 1, 0, -1):
        lines.append(rf"\subsection*{{Система {i}}}")
        nx, ny, nz = sp.symbols(f'n_{i}x n_{i}y n_{i}z')
        lines.append(r"\textbf{Момент:}")
        lines.append(r"\[" + sp.latex(sp.Eq(nx, sp.simplify(moments[i][0]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(ny, sp.simplify(moments[i][1]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(nz, sp.simplify(moments[i][2]))) + r"\]")
        fx, fy, fz = sp.symbols(f'f_{i}x f_{i}y f_{i}z')
        lines.append(r"\textbf{Сила:}")
        lines.append(r"\[" + sp.latex(sp.Eq(fx, sp.simplify(forces[i][0]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(fy, sp.simplify(forces[i][1]))) + r"\]")
        lines.append(r"\[" + sp.latex(sp.Eq(fz, sp.simplify(forces[i][2]))) + r"\]")
        lines.append("")

    tex_content = (
        r"\documentclass{article}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[russian]{babel}" + "\n"
        r"\usepackage{amsmath}" + "\n"
        r"\usepackage{geometry}" + "\n"
        r"\geometry{margin=2cm}" + "\n"
        r"\begin{document}" + "\n\n"
        + "\n".join(lines) + "\n\n"
        r"\end{document}"
    )

    with open("dynamics_output.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)

    print("\nРезультаты записаны в dynamics_output.tex")