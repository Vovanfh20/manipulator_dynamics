import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

from Robot_parameters import (
    dh_params,
    default_theta_point, default_d_point,
    default_theta_ddot, default_d_ddot,
    P_Center_coordinates, g
)
from Robot_velocities import Manipulator_kinematics


class Link:
    """Звено манипулятора. DH-параметры + центр масс."""

    def __init__(self, a, alpha, d, theta, p_center):
        self.a        = a
        self.alpha    = alpha
        self.d        = d
        self.theta    = theta
        self.p_center = p_center
    """Матрица поворота."""     
    def R_rotation(self):
        theta, a = self.theta, self.alpha
        return sp.Matrix([
            [sp.cos(theta),              -sp.sin(theta),              0         ],
            [sp.sin(theta)*sp.cos(a),    sp.cos(theta)*sp.cos(a),   -sp.sin(a) ],
            [sp.sin(theta)*sp.sin(a),    sp.cos(theta)*sp.sin(a),    sp.cos(a) ],
        ])
    """Вектор P."""
    def P_coordinates(self):
        return sp.Matrix([
            self.a,
            -self.d * sp.sin(self.alpha),
             self.d * sp.cos(self.alpha)
        ])
    """Координаты центров масс, берутся из файлв Robot_parameters"""
    def P_center(self):
        return sp.Matrix(list(self.p_center))
    """Угловая скорость вращательного сочленения вдоль z."""
    def dot_theta_Z(self, dot_theta):
        
        return sp.Matrix([0, 0, dot_theta])
    """Линейная скорость поступательного сочленения вдоль z."""
    def dot_d_Z(self, dot_d):
        
        return sp.Matrix([0, 0, dot_d])
    """Угловое ускорение вращательного сочленения вдоль z."""
    def ddot_theta_Z(self, ddot_theta):
        
        return sp.Matrix([0, 0, ddot_theta])
    """Линейное ускорение поступательного сочленения вдоль z."""
    def ddot_d_Z(self, ddot_d):
        
        return sp.Matrix([0, 0, ddot_d])
   
class Manipulator_Acceleration:
    """Расчёт ускорений звеньев."""

    def __init__(self):
        self.links = []

    def add_link(self, a, alpha, d, theta, p_center):
        self.links.append(Link(a, alpha, d, theta, p_center))

    @classmethod
    def from_dh_params(cls, dh_params, p_center_list):
        assert len(dh_params) == len(p_center_list), (
        )
        robot = cls()
        for (a, alpha, d, theta), p_center in zip(dh_params, p_center_list):
            robot.add_link(a, alpha, d, theta, p_center)
        return robot

    def forward_acceleration(self,
                             omega_list,
                             ddot_theta_list,
                             ddot_d_list,
                             dot_theta_list,
                             dot_d_list):
        """
        Итерационный расчёт ускорений.

        Параметры:
            omega_list      - угловые скорости из forward_kinematics
            ddot_theta_list - угловые ускорения сочленений
            ddot_d_list     -линейные ускорения сочленений
            dot_theta_list  - угловые скорости сочленений
            dot_d_list      - линейные скорости сочленений

        Возвращает:
            epsilon_list - угловые ускорения [ε_0..ε_n] в СК звена {i}
            a_list       - линейные ускорения [a_0..a_n] в СК звена {i}
            a_c_list     —-ускорения центров масс [ac_1..ac_n] в СК звена {i}
        """
        n = len(self.links)

        epsilon_list = [sp.Matrix([0, 0, 0]) for _ in range(n + 1)]
        a_list       = [sp.Matrix([0, 0, 0]) for _ in range(n + 1)]
        a_c_list     = [sp.Matrix([0, 0, 0]) for _ in range(n + 1)]

        # Начальные условия — трюк Крейга для учёта гравитации
        epsilon_list[0] = sp.Matrix([0, 0, 0])
        a_list[0]       = sp.Matrix([0, 0, g])

        for i, link in enumerate(self.links):
            R_T = link.R_rotation().T
            P   = link.P_coordinates()
            Pc  = link.P_center()

            omi  = omega_list[i]    # угловая скорость текущего звена {i}
            ei  = epsilon_list[i]  # угловое ускорение текущего звена {i}
            ai  = a_list[i]        # линейное ускорение начала СК {i}

            # --- Угловое ускорение (Craig eq. 6.45) ---
            # Для вращательного: добавляем ddot_theta, для призматического: нули
            epsilon_list[i+1] = (R_T * ei+ (R_T * omi).cross(link.dot_theta_Z(dot_theta_list[i]))+ link.ddot_theta_Z(ddot_theta_list[i])
            )

            # --- Линейное ускорение начала СК {i+1} (Craig eq. 6.46) ---
            # Для призматического: добавляем 2*omega x d_dot и ddot_d
            a_list[i+1] = R_T * (ei.cross(P)+ omi.cross(omi.cross(P))+ ai) + 2 * omega_list[i+1].cross(link.dot_d_Z(dot_d_list[i]))+ link.ddot_d_Z(ddot_d_list[i])

            # --- Ускорение центра масс звена {i+1} (Craig eq. 6.47) ---
            # Вычисляется в СК {i+1}, поэтому используем epsilon и omega [i+1]
            a_c_list[i + 1] = epsilon_list[i + 1].cross(Pc)+ omega_list[i + 1].cross(omega_list[i + 1].cross(Pc)) + a_list[i + 1]
            

        return epsilon_list, a_list, a_c_list

    
    def Rotation_total(self):
        """Итоговая матрица поворота R = R1 * R2 * ... * Rn."""
        R_total = sp.eye(3)
        for link in self.links:
            R_total = R_total * link.R_rotation()
        return R_total

if __name__ == "__main__":

    # --- 1. Сначала считаем кинематику (нужен omega_list) ---
    kin = Manipulator_kinematics.from_dh_params(dh_params)
    n = len(kin.links)
    omega_list, v_list = kin.forward_kinematics(default_theta_point, default_d_point)

    # --- 2. Создаём модель ускорения ---
    acc = Manipulator_Acceleration.from_dh_params(dh_params, P_Center_coordinates)

    epsilon_list, a_list, a_c_list = acc.forward_acceleration(
        omega_list      = omega_list,
        ddot_theta_list = default_theta_ddot,
        ddot_d_list     = default_d_ddot,
        dot_theta_list  = default_theta_point,
        dot_d_list      = default_d_point,
    )

    R_final = acc.Rotation_total()

    # --- 3. Вывод угловых ускорений каждого звена ---
    print("УГЛОВЫЕ УСКОРЕНИЯ ЗВЕНЬЕВ (в СК звена {i})")
    for i in range(1, n + 1):
        ep = sp.simplify(sp.trigsimp(epsilon_list[i]))
        ex, ey, ez = sp.symbols(f'epsilon_{i}x epsilon_{i}y epsilon_{i}z')
        display(sp.Eq(ex, ep[0]))
        display(sp.Eq(ey, ep[1]))
        display(sp.Eq(ez, ep[2]))

    # --- 4. Вывод линейных ускорений каждого сочленения ---
    print("ЛИНЕЙНЫЕ УСКОРЕНИЯ СОЧЛЕНЕНИЙ (в СК звена {i})")
    for i in range(1, n + 1):
        a = sp.simplify(sp.trigsimp(a_list[i]))
        ax, ay, az = sp.symbols(f'a_{i}x a_{i}y a_{i}z')
        display(sp.Eq(ax, a[0]))
        display(sp.Eq(ay, a[1]))
        display(sp.Eq(az, a[2]))

    # --- 5. Вывод ускорений центров масс ---
    print("УСКОРЕНИЯ ЦЕНТРОВ МАСС ЗВЕНЬЕВ (в СК звена {i})")
    for i in range(1, n + 1):
        ac = sp.simplify(sp.trigsimp(a_c_list[i]))
        acx, acy, acz = sp.symbols(f'ac_{i}x ac_{i}y ac_{i}z')
        display(sp.Eq(acx, ac[0]))
        display(sp.Eq(acy, ac[1]))
        display(sp.Eq(acz, ac[2]))

     # --- 6. Ускорение схвата в базовой СК ---
    print("УСКОРЕНИЕ СХВАТА В БАЗОВОЙ СК {0}")
    a_end_base = sp.simplify(sp.trigsimp(acc.Rotation_total() * a_list[n]))  
    ax_s = sp.Symbol('a_{end,x}')
    ay_s = sp.Symbol('a_{end,y}')
    az_s = sp.Symbol('a_{end,z}')
    display(sp.Eq(ax_s, a_end_base[0]))
    display(sp.Eq(ay_s, a_end_base[1]))
    display(sp.Eq(az_s, a_end_base[2]))
    # ── 7. Экспорт результатов в LaTeX ───────────────────────────────────
    lines = []
    
    # ── вспомогательные функции для LaTeX ────────────────────────────────
    def _latex_vector_full(vec, name):
        """Вектор целиком в окружении pmatrix."""
        return r"\[" + name + r" = " + sp.latex(vec, mat_str='pmatrix') + r"\]"
    
    def _latex_vector_elements(vec, name):
        """Каждый ненулевой элемент вектора отдельной строкой."""
        out = []
        for i in range(vec.shape[0]):
            elem = sp.simplify(sp.trigsimp(vec[i]))
            if elem == 0:
                continue
            lhs_sym = sp.Symbol(f'{name}_{{{i + 1}}}')
            out.append(r"\[" + sp.latex(sp.Eq(lhs_sym, elem)) + r"\]")
        return out
    
    # ── заголовок ────────────────────────────────────────────────────────
    lines.append(r"\section*{Ускорения звеньев манипулятора}")
    lines.append(r"Пусть $n = " + str(n) + r"$ — число звеньев.")
    lines.append("")
    
    # ── угловые ускорения ────────────────────────────────────────────────
    lines.append(r"\section*{Угловые ускорения $\epsilon_i$}")
    for i in range(1, n + 1):
        lines.append(rf"\subsection*{{Звено {i}}}")
        lines.append(_latex_vector_full(sp.simplify(sp.trigsimp(epsilon_list[i])), f"\epsilon_{i}"))
        lines.extend(_latex_vector_elements(epsilon_list[i], f"\epsilon_{i}"))
        lines.append("")
    
    # ── линейные ускорения ───────────────────────────────────────────────
    lines.append(r"\section*{Линейные ускорения $a_i$}")
    for i in range(1, n + 1):
        lines.append(rf"\subsection*{{Звено {i}}}")
        lines.append(_latex_vector_full(sp.simplify(sp.trigsimp(a_list[i])), f"a_{i}"))
        lines.extend(_latex_vector_elements(a_list[i], f"a_{i}"))
        lines.append("")
    
    # ── ускорения центров масс ───────────────────────────────────────────
    lines.append(r"\section*{Ускорения центров масс $a_{c_i}$}")
    for i in range(1, n + 1):
        lines.append(rf"\subsection*{{Звено {i}}}")
        lines.append(_latex_vector_full(sp.simplify(sp.trigsimp(a_c_list[i])), f"a_{{c_{i}}}"))
        lines.extend(_latex_vector_elements(a_c_list[i], f"a_{{c_{i}}}"))
        lines.append("")
    
    # ── ускорение схвата в базовой СК ───────────────────────────────────
    lines.append(r"\section*{Ускорение схвата в базовой СК}")
    lines.append(_latex_vector_full(sp.simplify(sp.trigsimp(a_end_base)), "a_{end}"))
    lines.extend(_latex_vector_elements(a_end_base, "a_{end}"))
    lines.append("")
    
    # ── сборка .tex ──────────────────────────────────────────────────────
    tex_content = (
        r"\documentclass{article}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[T2A]{fontenc}" + "\n"
        r"\usepackage[russian]{babel}" + "\n"
        r"\usepackage{amsmath}" + "\n"
        r"\usepackage{amssymb}" + "\n"
        r"\usepackage{geometry}" + "\n"
        r"\geometry{margin=2cm}" + "\n"
        r"\begin{document}" + "\n\n"
        + "\n".join(lines) + "\n\n"
        r"\end{document}"
    )
    
    with open("acceleration_output.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)
    
    print("\nРезультаты записаны в acceleration_output.tex")