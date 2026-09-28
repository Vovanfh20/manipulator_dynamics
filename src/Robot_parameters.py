# Robot_parameters.py
import sympy as sp

# Геометрические параметры
# positive=True — длины звеньев положительны (помогает sympy упрощать)
# real=True     — углы вещественные (без этого sympy даёт комплексные решения!)
L, L1, L2, L3 = sp.symbols('L L_{1} L_{2} L_{3}', positive=True)
th1, th2, th3, th4, th5, th6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6', real=True)
d1, d2, d3, d4, d5, d6 = sp.symbols('d1 d2 d3 d4 d5 d6', real=True)
R = sp.symbols('R', real=True) #Радиус цилиндра

# Скорости 
dot_theta1, dot_theta2, dot_theta3, dot_theta4, dot_theta5, dot_theta6 = sp.symbols(r'\dot{\theta}_1 \dot{\theta}_2 \dot{\theta}_3 \dot{\theta}_4 \dot{\theta}_5 \dot{\theta}_6', real=True)
dot_d1, dot_d2, dot_d3 = sp.symbols(r'\dot{d}_1 \dot{d}_2 \dot{d}_3', real=True)

# Ускорения 
ddot_theta1, ddot_theta2, ddot_theta3 = sp.symbols(r'\ddot{\theta}_1 \ddot{\theta}_2 \ddot{\theta}_3', real=True)
ddot_d1, ddot_d2, ddot_d3 = sp.symbols(r'\ddot{d}_1 \ddot{d}_2 \ddot{d}_3', real=True)

#Динамика 
m, m1, m2, m3 = sp.symbols('m m_{1} m_{2} m_{3}', positive=True)

# ─── Компоненты тензоров инерции ─────────────────────────────────────────────
Ixx1, Iyy1, Izz1, Ixy1, Ixz1, Iyz1 = sp.symbols('I_{xx1} I_{yy1} I_{zz1} I_{xy1} I_{xz1} I_{yz1}', real=True)
Ixx2, Iyy2, Izz2, Ixy2, Ixz2, Iyz2 = sp.symbols('I_{xx2} I_{yy2} I_{zz2} I_{xy2} I_{xz2} I_{yz2}', real=True)
Ixx3, Iyy3, Izz3, Ixy3, Ixz3, Iyz3 = sp.symbols('I_{xx3} I_{yy3} I_{zz3} I_{xy3} I_{xz3} I_{yz3}', real=True)


g = sp.symbols('g')

# Переменные для вывода
x_symb, y_symb, z_symb = sp.symbols('x y z')
v_x_symb, v_y_symb, v_z_symb = sp.symbols(r'V_{x} V_{y} V_{z}')
w_x_symb, w_y_symb, w_z_symb = sp.symbols(r'W_{x} W_{y} W_{z}' )



# DH-параметры: (a, alpha, d, theta)
dh_params = [
    (0,  0, 0, th1),       # звено 1 
    (L1, 0, 0, th2),       # звено 2
    (L2, 0, -d3, 0),       # звено 3 
     
]

# Скорости сочленений 
default_d_point     = [0, 0, dot_d3]
default_theta_point = [dot_theta1, dot_theta2, 0]


# Обобщённые координаты и скорости для якобиана
default_q     = [th1, th2, d3]
default_q_dot = [dot_theta1, dot_theta2, dot_d3]
det_J_symb    = sp.Symbol(r'\det(J)')

#─── Параметры для статики  ───────────────────────────────────────────────────────────
#Типы суставов ДЛЯ СТАТИКИ 
# 'R' — revolute (вращательный), момент τ = z^T · n
# 'P' — prismatic (призматический), сила  τ = z^T · f\
# от базы (стойки) до схвата, заполняем по порядку
joint_types = ['R', 'R', 'P']

#─── Силы и моменты на концевом эффекторе (в кадре {N}) ─────────────────────
# Используются в задаче статики как нагрузка на схвате
f_x, f_y, f_z = sp.symbols(r'f_{x} f_{y} f_{z}')
n_x, n_y, n_z = sp.symbols(r'n_{x} n_{y} n_{z}')

#─── Параметры для ускорений  ───────────────────────────────────────────────────────────
#Координаты для центра масс
P_Center_coordinates = [
    (L1/2, 0, 0),
    (L2/2, 0, 0),
    (0, 0, d3/2)
    ]
"""
Данные для статики
"""

# Ускорения сочленений
default_d_ddot     = [0, 0, ddot_d3]
default_theta_ddot = [ddot_theta1, ddot_theta2, 0]


#─── Параметры для динамики───────────────────────────────────────────────────────────
masses = [m1, m2, m3]

# Для расчета матриц, массив обобщённых координат:
q_symbols = [th1, th2, d3]

# ─── Тензоры инерции 3x3 (симметричные матрицы) ──────────────────────────────
I1 = sp.Matrix([
    [m1*(R**2/4 + L1**2/12), 0, 0],
    [0, m1*(R**2/4 + L1**2/3), 0],
    [0, 0, m1*(R**2/2 + L1**2/4)]
])

I2 = sp.Matrix([
    [m2*(R**2/4 + L2**2/12), 0, 0],
    [0, m2*(R**2/4 + L2**2/3), 0],
    [0, 0, m2*(R**2/2 + L2**2/4)]
])

"""
# Временно делаем "испорченный" тензор:
I2 = sp.Matrix([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
"""
I3 = sp.Matrix([
    [m3*(R**2/4 + L3**2/3), 0, 0],
    [0, m3*(R**2/4 + L3**2/3), 0],
    [0, 0, m3*R**2/2]
])


# ─── Массив тензоров (индекс 0 → звено 1, и т.д.) ────────────────────────────
inertia_tensors = [sp.zeros(3), sp.zeros(3), sp.zeros(3)]














































































