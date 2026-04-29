import numpy as np
import matplotlib.pyplot as plt

M, m, l, g = 0.2, 0.1, 0.5, 9.81
b, c = 0.1, 0.01

A = np.array([
    [0, 1, 0, 0],
    [0, -b/M, -m*g/M, c/(M*l)],
    [0, 0, 0, 1],
    [0, b/(M*l), (M+m)*g/(M*l), -(M+m)*c/(M*m*l**2)]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*l)]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

Co = np.hstack([B, A @ B, np.linalg.matrix_power(A, 2) @ B, np.linalg.matrix_power(A, 3) @ B])
rank_Co = np.linalg.matrix_rank(Co)

print(f"Ранг матриці керованості: {rank_Co}")
if rank_Co == 4:
    print("Система повністю керована")
Ob = np.vstack([C, C @ A, C @ np.linalg.matrix_power(A, 2), C @ np.linalg.matrix_power(A, 3)])
rank_Ob = np.linalg.matrix_rank(Ob)

print(f"Ранг матриці спостережуваності: {rank_Ob}")
if rank_Ob == 4:
    print("Система повністю спостережувана")

A_eigen_values = np.linalg.eigvals(A)

print("\nВласні значення матриці А:")
is_stable = True
for val in A_eigen_values:
    print(f"λ = {val.real:.4f} + {val.imag:.4f}j")
    if val.real > 0:
        is_stable = False
if is_stable:
    print("Висновок: Система стійка")
else:
    print("Висновок: Система нестійка (має власні значення у правій півплощині) і зростатиме експоненційно")

real_parts = [val.real for val in A_eigen_values]
imag_parts = [val.imag for val in A_eigen_values]

plt.figure(figsize=(10, 7))

plt.scatter(real_parts, imag_parts, color='red', marker='x', s=100, label='Власні значення (Poles)')

plt.axhline(0, color='black', linestyle='-', linewidth=1) # Вісь Real
plt.axvline(0, color='black', linestyle='-', linewidth=1) # Вісь Imaginary

plt.grid(True, linestyle='--', alpha=0.7)

limit = max(max(np.abs(real_parts)), max(np.abs(imag_parts)), 1) * 1.2
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.title('Розташування власних значень у комплексній площині', fontsize=14)
plt.xlabel('Re (Дійсна частина)', fontsize=12)
plt.ylabel('Im (Уявна частина)', fontsize=12)
plt.legend()
plt.text(limit*0.1, limit*0.1, 'Нестійка зона (Re > 0)', color='red', alpha=0.5, fontsize=10)
plt.text(-limit*0.9, limit*0.1, 'Стійка зона (Re < 0)', color='green', alpha=0.5, fontsize=10)

plt.show()