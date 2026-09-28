# Символьная кинематика и динамика SCARA-манипулятора

Вычислительное ядро дипломной работы: **символьный (SymPy) расчёт кинематики, статики и динамики
трёхстепенного SCARA-манипулятора** (два вращательных сустава + один призматический, конфигурация
RRP). Все величины выводятся в аналитическом виде и экспортируются в LaTeX.

> **Short summary (EN).** Symbolic modeling of a 3-DOF SCARA manipulator (RRP) with SymPy: forward
> kinematics from Denavit–Hartenberg parameters, link velocities via Craig's iterative method, the
> geometric Jacobian with singularity analysis, static force/torque propagation, link accelerations and
> the recursive **Newton–Euler dynamics**, from which the manipulator equation `τ = M(q)·q̈ +
> C(q,q̇)·q̇ + G(q)` is extracted. The structural properties of `M` and `C` are then verified
> symbolically. Each stage exports its results to LaTeX.

Работа опирается на итеративные формулы Крейга (J. J. Craig, *Introduction to Robotics: Mechanics and
Control*). Полный текст диплома — в [`docs/Диплом_Гефлинг_В.И.pdf`](docs/Диплом_Гефлинг_В.И.pdf).

## Модель робота

SCARA задан DH-параметрами (файл `Robot_parameters.py`): два вращательных сустава `θ₁, θ₂` и
призматический `d₃`. Все параметры (длины звеньев, массы, тензоры инерции, координаты центров масс,
типы суставов) собраны в одном месте — остальные модули берут их оттуда.

Результаты согласуются с классической моделью SCARA, например:

```
Прямая кинематика:  x = L₁cosθ₁ + L₂cos(θ₁+θ₂),  y = L₁sinθ₁ + L₂sin(θ₁+θ₂),  z = −d₃
Якобиан:            det J = −L₁L₂ sin θ₂   →   сингулярности при θ₂ = 0, π
Свойства динамики:  M(q) = Mᵀ ≻ 0,   N = Ṁ − 2C — кососимметрична (N + Nᵀ = 0)
```

## Порядок расчёта (конвейер)

Модули задуманы как последовательность; `Robot_parameters.py` — всегда первый, остальные берут данные
из него.

1. **`Robot_parameters.py`** — параметры робота: DH-таблица, скорости/ускорения сочленений, массы,
   тензоры инерции, центры масс, типы суставов. Импортируется всеми остальными.
2. **`Robot_geometry.py`** — прямая задача кинематики: матрицы Денавита–Хартенберга и итоговое
   преобразование `T = T₁·T₂·T₃`, координаты схвата.
3. **`Robot_velocities.py`** — угловые и линейные скорости звеньев (итеративный метод Крейга).
4. **`Robot_statics.py`** — статика: перенос сил и моментов от схвата к базе, обобщённые силы в суставах.
5. **`Robot_accelerations.py`** — угловые и линейные ускорения звеньев и ускорения центров масс
   (прямой ход по Крейгу).
6. **`Robot_dynamics.py`** — динамика Ньютона–Эйлера (обратный ход): обобщённые силы `τ`, из которых
   извлекаются матрица масс `M`, матрица Кориолиса `C` (через символы Кристоффеля) и вектор гравитации `G`.
7. **`Проверка свойств матриц.py`** — символьная проверка структурных свойств: `M(q)` положительно
   определённая (`M = Mᵀ ≻ 0`) и `N = Ṁ − 2C` кососимметрична (`N + Nᵀ = 0`).

Дополнительно: **`Jacobi_calculations.py`** — матрица Якоби `J = ∂P/∂q`, скорость схвата `v = J·q̇`,
обобщённые скорости `q̇ = J⁻¹·v` и точки сингулярности `det J = 0`.

## Запуск

Расчёты используют SymPy и рассчитаны на вывод в Jupyter (`IPython.display`); каждый модуль также
пишет результат в `.tex`. Пример:

```bash
pip install sympy ipython
cd src
python Robot_geometry.py            # прямая кинематика
python Robot_velocities.py          # скорости
python Jacobi_calculations.py       # якобиан и сингулярности
python Robot_statics.py             # статика
python Robot_accelerations.py       # ускорения
python "Проверка свойств матриц.py"  # динамика + проверка свойств M, C
```

## Литература

1. J. J. Craig. *Introduction to Robotics: Mechanics and Control.* 3rd ed. — Pearson, 2005.
2. Курс лекций по введению в робототехнику (итеративные методы кинематики и динамики манипуляторов).

---

**Автор:** Гефлинг В. И.
