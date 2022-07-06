import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def wilkinson_generator(dimension, c_value):
    matrix = np.zeros((dimension, dimension))
    # value = int(dimension / 2)
    # matrix[1][0] = 1
    # for i in range(1, dimension - 1):
    #     matrix[i - 1][i] = 1
    #     matrix[i + 1][i] = 1
    #
    # for i in range(int(dimension / 2) + 1):
    #     matrix[i][i] += value
    #     matrix[dimension - i - 1][dimension - i - 1] = 1
    #     value -= 1

    for i in range(dimension):
        matrix[i][i] = 1.0
        matrix[i][dimension - 1] = 1.0
        for j in range(i + 1, dimension):
            matrix[j][i] = c_value

    random_x = np.round(np.random.uniform(1, dimension, (dimension, 1)))

    b = matrix.dot(random_x)

    return matrix, random_x, b


def generate_test(min_value, max_value, dimension):
    random_A = np.round(np.random.uniform(min_value, max_value, (dimension, dimension)), 3)
    random_x = np.round(np.random.uniform(min_value, max_value, (dimension, 1)), 3)
    random_b = random_A.dot(random_x)
    # random_A = np.append(random_A, random_b, axis=1)
    return random_A, random_x, random_b


def bubble_max_row(m, col):
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        m[[col, max_row]] = m[[max_row, col]]


# with the choice of the main element by column
def solve_gauss(m, b):
    start_time = datetime.now()
    m = np.append(m, b, axis=1)
    n = len(m)
    for i in range(n - 1):
        bubble_max_row(m, i)
        for j in range(i + 1, n):
            div = m[j][i] / m[i][i]
            m[j][i: n + 1] -= div * m[i, i: n + 1]

    # backward trace
    answer = np.zeros(n)
    for k in range(n - 1, -1, -1):
        answer[k] = (m[k][-1] - sum([m[k][j] * answer[j] for j in range(k + 1, n)])) / m[k][k]

    time = datetime.now() - start_time
    return answer, time.total_seconds(), np.linalg.cond(m)


def householder(A, b):
    time_start = datetime.now()
    n = len(A)
    alpha = np.zeros(n)
    for j in range(0, n):
        alpha[j] = np.linalg.norm(A[j:, j]) * np.sign(A[j, j])
        if alpha[j]:
            beta = 1 / math.sqrt(2 * alpha[j] * (alpha[j] + A[j, j]))
            A[j, j] = beta * (A[j, j] + alpha[j])
            A[j + 1:, j] = beta * A[j + 1:, j]
            for k in range(j + 1, n):
                gamma = 2 * A[j:, j].dot(A[j:, k])
                A[j:, k] = A[j:, k] - gamma * A[j:, j]
    answer = loese_householder(A, alpha, b)
    time = datetime.now() - time_start
    return answer, time.total_seconds()


def loese_householder(H, alpha, b):
    (n, m) = H.shape
    b = b.copy()
    x = np.zeros(n)
    # b=Q^t b.
    for j in range(0, n):
        b[j:] = b[j:] - 2 * (H[j:, j].dot(b[j:])) * H[j:, j]

    for i in range(0, n):
        j = n - 1 - i
        b[j] = b[j] - H[j, j + 1:].dot(x[j + 1:])
        x[j] = -b[j] / alpha[j]
    return x


def standart_gauss(a, b):
    time_start = datetime.now()
    answer = np.linalg.solve(a, b);
    time = datetime.now() - time_start
    return answer, time.total_seconds()


def draw(x_array, y_array, x_label, y_label):
    plt.plot(x_array, y_array, color = "red")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



error_array_gauss = []
error_array_householder = []
time_array_gauss = []
time_array_householder = []
time_array_standart_gauss = []
dim_array = []
cond_array = []
# c_array = []
c_temp = 0
# while c_temp <= 1:
#     c_array.append(c_temp)
#     c_temp += 0.05

c_array = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
# initial value of dimension and c
dim = 200
c_value = 0.99
np.set_printoptions(precision=5)
for i in range(len(c_array)):  # set max dimension
#while(dim <= 500):
    # matrix, x, b = generate_test(1, 10, dim)
    matrix, x, b = wilkinson_generator(dim, -c_array[i])

    # gauss
    answer_gauss, time_gauss, cond_value = solve_gauss(matrix.copy(), b.copy())
    cond_array.append(cond_value)
    # print(matrix)
    # print(b.transpose()[0])
    b = np.transpose(b)
    b = b[0]

    # householder
    answer_householder, time_householder = householder(matrix.copy(), b.copy())
    time_array_gauss.append(time_gauss)
    time_array_householder.append(time_householder)

    # print(b)
    dim_array.append(dim)
    # error_array_gauss.append(np.linalg.norm(np.subtract(matrix.dot(answer_gauss), b), ord = 2))
    # error_array_householder.append(np.linalg.norm(np.subtract(matrix.dot(answer_householder), b), ord = 2))
    error_array_gauss.append(np.linalg.norm(np.subtract(answer_gauss, x.transpose())))
    error_array_householder.append(np.linalg.norm(np.subtract(answer_householder, x.transpose())))
    # print(x.transpose())
    # print(answer_householder)
    # set dimension step
    # dim += 20
    print(c_array[i])
    # print(c_array[i])

    # print(error_array_gauss)
    # print(error_array_householder)

info = {'c value' : c_array, 'cond after Gauss' : cond_array, 'residual norm Gauss': error_array_gauss}

print(tabulate(info, headers ='keys'))

plt.plot(c_array, error_array_gauss, color = "orange")
plt.plot(c_array, error_array_householder, color = "black")
plt.xlabel("c value")
plt.ylabel("residual norm")
plt.legend(["Gauss", "Householder"])
plt.yscale('log')
plt.show()

