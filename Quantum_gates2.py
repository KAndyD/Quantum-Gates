import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Визуализация одиночного кубита ==========
def visualize_qubit(qubit, title="Состояние кубита"):
    """
    Визуализирует вероятность нахождения кубита в состоянии |0⟩ и |1⟩.
    """
    probabilities = np.abs(qubit.state) ** 2
    labels = ['|0⟩', '|1⟩']

    plt.bar(labels, probabilities, color=['#4A90E2', '#50E3C2'])
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Вероятность')
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()


# ========== 2. Визуализация пары кубитов ==========
def visualize_pair(state_vector, title="Состояние пары кубитов"):
    """
    Визуализирует вероятности состояний |00⟩, |01⟩, |10⟩, |11⟩.
    """
    probabilities = np.abs(state_vector) ** 2
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']

    plt.bar(labels, probabilities, color=['#4A90E2', '#50E3C2', 'orange', '#AD66D5'])
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Вероятность')
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()


# ========== 3. Класс Qubit ==========
class Qubit:
    def __init__(self, alpha=1.0, beta=0.0):
        """
        Инициализирует кубит в состоянии |ψ⟩ = α|0⟩ + β|1⟩.
        """
        self.state = np.array([complex(alpha), complex(beta)], dtype=complex)
        self.normalize()

    def normalize(self):
        """
        Обеспечивает нормализацию: |α|² + |β|² = 1.
        """
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Нулевой вектор состояния недопустим.")
        self.state /= norm

    def apply_gate(self, gate_matrix):
        """
        Применяет квантовый гейт к текущему состоянию.
        """
        self.state = np.dot(gate_matrix, self.state)

    def measure(self):
        """
        Выполняет квантовое измерение: возвращает 0 или 1 на основе вероятностей.
        """
        probabilities = np.abs(self.state) ** 2
        return np.random.choice([0, 1], p=probabilities)

    def __str__(self):
        """
        Возвращает строковое представление состояния кубита.
        """
        return f"{self.state[0]:.2f}|0⟩ + {self.state[1]:.2f}|1⟩"


# ========== 4. Гейты Паули ==========
def pauli_x():
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def pauli_y():
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)

def pauli_z():
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)

# ========== 5. Гейт Адамара ==========
def hadamard():
    """
    Гейт Адамара: создает суперпозицию состояний.
    """
    return (1 / np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex)


# ========== 6. Гейт CNOT (двухкубитный) ==========
def cnot():
    """
    Гейт CNOT: controlled-NOT, применяется к паре кубитов.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


# ========== 7. Класс пары кубитов ==========
class QubitPair:
    def __init__(self, q1: Qubit, q2: Qubit):
        """
        Инициализирует пару кубитов (тензорное произведение состояний).
        """
        self.state = np.kron(q1.state, q2.state)

    def apply_gate(self, gate_matrix):
        """
        Применяет двухкубитный гейт (матрица 4x4).
        """
        self.state = np.dot(gate_matrix, self.state)

    def __str__(self):
        """
        Возвращает строковое представление двухкубитного состояния.
        """
        return (f"{self.state[0]:.2f}|00⟩ + {self.state[1]:.2f}|01⟩ + "
                f"{self.state[2]:.2f}|10⟩ + {self.state[3]:.2f}|11⟩")


# ==========================
# 🔬 Примеры демонстрации
# ==========================

# ▶ Пример 1: Изначальное состояние и X/Y/Z-гейты
q = Qubit(1, 0)  # |0⟩
print("Начальное состояние:", q)
visualize_qubit(q, "Начальное состояние |0⟩")

q.apply_gate(pauli_x())
print("После X-гейта:", q)
visualize_qubit(q, "После X-гейта")

q.apply_gate(pauli_y())
print("После Y-гейта:", q)
visualize_qubit(q, "После Y-гейта")

q.apply_gate(pauli_z())
print("После Z-гейта:", q)
visualize_qubit(q, "После Z-гейта")


# ▶ Пример 2: Суперпозиция через H
q_super = Qubit(1, 0)           # |0⟩
q_super.apply_gate(hadamard()) # H|0⟩ → суперпозиция
print("Суперпозиция после H:", q_super)
visualize_qubit(q_super, "Кубит после H (суперпозиция)")


# ▶ Пример 3: Генерация запутанного состояния
q1 = Qubit(1, 0)                # Первый кубит |0⟩
q2 = Qubit(1, 0)                # Второй кубит |0⟩

q1.apply_gate(hadamard())      # H на первый → суперпозиция

pair = QubitPair(q1, q2)       # Создаем пару
pair.apply_gate(cnot())        # CNOT → запутывание
print("Запутанное состояние (Bell):", pair)
visualize_pair(pair.state, "Запутанное состояние Bell: (|00⟩ + |11⟩)/√2")