# Robot_properties.py
"""
Проверка свойств уравнения динамики манипулятора:
    M(q)·q̈ + C(q,q̇)·q̇ + G(q) = τ

1°. M(q) — положительно определённая:  M = Mᵀ > 0
2°. N := Ṁ(q) − 2C(q,q̇) — кососимметрическая: N + Nᵀ = 0
"""

import sympy as sp
from IPython.display import display

sp.init_printing(use_latex=True)

# ── Импорт результатов динамики ───────────────────────────────────────────────
from Robot_parameters import (
    dh_params,
    joint_types,
    masses,
    inertia_tensors,
    P_Center_coordinates,
    q_symbols,
    default_theta_point, default_d_point,
    default_theta_ddot,  default_d_ddot, dot_theta1, dot_theta2, dot_d3, ddot_theta1, 
    ddot_theta2, ddot_d3, 
)
from Robot_velocities    import Manipulator_kinematics
from Robot_accelerations import Manipulator_Acceleration
from Robot_dynamics      import Robot_dynamics


# ══════════════════════════════════════════════════════════════════════════════
#  Вспомогательная функция: цветной вывод результата проверки
# ══════════════════════════════════════════════════════════════════════════════
def _print_result(label, passed):
    mark = "ВЫПОЛНЕНО" if passed else "НЕ ВЫПОЛНЕНО"
    print(f"  {mark}  — {label}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Получаем M и C из полного расчёта динамики
# ══════════════════════════════════════════════════════════════════════════════
def compute_MCG():
    """Запускает полный конвейер кинематика→ускорения→динамика и возвращает M, C, G."""
    

    q_dot_syms  = [dot_theta1, dot_theta2, dot_d3]
    # Для q'' аналогично
    q_ddot_syms = [ddot_theta1, ddot_theta2, ddot_d3]

    print("  Кинематика...")
    kin = Manipulator_kinematics.from_dh_params(dh_params)
    omega_list, v_list = kin.forward_kinematics(default_theta_point, default_d_point)

    print("  Ускорения...")
    acc = Manipulator_Acceleration.from_dh_params(dh_params, P_Center_coordinates)
    epsilon_list, a_list, a_c_list = acc.forward_acceleration(
        omega_list=omega_list,
        ddot_theta_list=default_theta_ddot,
        ddot_d_list=default_d_ddot,
        dot_theta_list=default_theta_point,
        dot_d_list=default_d_point,
    )

    print("  Динамика (Ньютон–Эйлер)...")
    dyn = Robot_dynamics.from_dh_params(dh_params, P_Center_coordinates)
    tau, _, _ = dyn.compute_dynamics(omega_list, epsilon_list, a_c_list)

    print("  Извлечение M, C, G...")
    M_mat, C_mat, h_vec, G_vec = Robot_dynamics.extract_MCG(
        tau, q_ddot_syms, q_dot_syms, q_list=q_symbols
    )

    return M_mat, C_mat, G_vec, q_dot_syms, q_ddot_syms


# ══════════════════════════════════════════════════════════════════════════════
#  2. Проверка: M положительно определённая
# ══════════════════════════════════════════════════════════════════════════════

def check_positive_definite(M_mat):
    """
    Символьная проверка положительной определённости M:
    пытаемся выполнить разложение Холецкого.
    """
    
    print("\n" + "═"*60)
    print("  ПРОВЕРКА 1:  M(q) — положительно определённая")
    print("═"*60)

    print("\n  [A] Символьное разложение Холецкого...")
    try:
        L_chol = M_mat.cholesky(hermitian=False)
        print("      Матрица L (разложение Холецкого M = L·Lᵀ):")
        display(L_chol)
        _print_result("разложение Холецкого найдено", True)
        _print_result("M положительно определённая", True)
        return True, L_chol
    except sp.matrices.MatrixError as e:
        print(f" Символьное разложение не удалось: {e}")
        _print_result("M положительно определённая", False)
        return False, None


# ══════════════════════════════════════════════════════════════════════════════
#  3. Проверка: N = Ṁ − 2C кососимметрическая
# ══════════════════════════════════════════════════════════════════════════════

def compute_M_dot(M_mat, q_symbols, q_dot_syms):
    """
    Ṁ(q) = dM/dt = Σ_k  ∂M/∂q_k · q̇_k   (цепное правило)
    """
    n = M_mat.shape[0]
    M_dot = sp.zeros(n, n)
    for q_k, qd_k in zip(q_symbols, q_dot_syms):
        dM_dqk = M_mat.diff(q_k)
        M_dot += dM_dqk * qd_k
    return sp.simplify(M_dot)


def check_skew_symmetric(M_mat, C_mat, q_symbols, q_dot_syms):
    """
    Проверяет: N := Ṁ − 2C кососимметрична, т.е. N + Nᵀ = 0.
    Символьная проверка.
    """
    print("\n" + "═"*60)
    print("  ПРОВЕРКА 2:  N := Ṁ(q) − 2C(q,q̇) — кососимметрическая")
    print("═"*60)

    if C_mat is None:
        print("  Матрица C не вычислена (нужен q_list). Проверка пропущена.")
        return False, None

    print("\n  Вычисляю Ṁ = dM/dt (цепное правило)...")
    M_dot = compute_M_dot(M_mat, q_symbols, q_dot_syms)
    print("  Ṁ:")
    display(M_dot)

    print("\n  Вычисляю N = Ṁ − 2C...")
    N_mat = sp.trigsimp(sp.simplify(M_dot - 2 * C_mat))
    print("  N = Ṁ − 2C:")
    display(N_mat)

    print("\n  Символьная проверка: N + Nᵀ = 0 ?")
    N_plus_NT = sp.trigsimp(sp.simplify(N_mat + N_mat.T))
    print("  N + Nᵀ =")
    display(N_plus_NT)

    skew_symbolic = N_plus_NT == sp.zeros(*N_plus_NT.shape)
    _print_result("N + Nᵀ = 0 (символьно)", skew_symbolic)

    print()
    _print_result("N кососимметрическая", skew_symbolic)
    return skew_symbolic, N_mat


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("  Расчёт динамики...")
    print("=" * 60)
    M_mat, C_mat, G_vec, q_dot_syms, q_ddot_syms = compute_MCG()

    print("\n  Матрица масс M:")
    display(M_mat)

    if C_mat is not None:
        print("\n  Матрица кориолиса C:")
        display(C_mat)

    pd_ok, L_chol = check_positive_definite(M_mat)
   
    sk_ok, N_mat = check_skew_symmetric(M_mat, C_mat, q_symbols, q_dot_syms)

    # ── Итоговая результаты ───────────────────────────────────────────────────
    print("  Итоговые рузультаты")
    _print_result("M(q) положительно определённая  [M = Mᵀ > 0]",       pd_ok)
    _print_result("N = Ṁ − 2C кососимметрическая  [N + Nᵀ = 0]",        sk_ok)
    print()
    