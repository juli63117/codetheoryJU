import numpy as np

S = [[1,1,1,0,1],
     [1,0,1,1,0],
     [0,1,0,1,1],
     [1,1,0,1,0]]

A = np.array(S)

def find_leading_row(A, row, col, rows):
    for r in range(row, rows):
        if A[r, col] != 0:
            return r
    return None

# 1.1
def REF(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    row = 0

    # Методом Гаусса 
    # 1. В первом столбце выбрать элемент, отличный от нуля (ведущий элемент). 
    # Строку с ведущим элементом (ведущая строка), если она не первая, переставить на место первой строки 
    # 2. Разделить все элементы ведущей строки на ведущий элемент 
    # 3. К каждой строке, расположенной ниже ведущей, прибавить ведущую строку, умноженную соответственно на такое число, 
    # чтобы элементы, стоящие под ведущим оказались равными нулю.
    for col in range(cols):
        if row >= rows:
            break
        
        leading_row = find_leading_row(A, row, col, rows)
        
        if leading_row is None:
            continue
        
        if leading_row != row:
            A[[row, leading_row]] = A[[leading_row, row]]
        
        leading_element = A[row, col]
        A[row] /= leading_element
        
        for r in range(row + 1, rows):
            A[r] = abs(A[r] - A[row] * A[r, col])
        
        row += 1

    return A

A_REF = REF(A)
print("Базис REF\n", A_REF[np.any(A_REF != 0, axis=1)])

# 1.2
def RREF(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    row = 0

    for col in range(cols):
        if row >= rows:
            break
        
        leading_row = find_leading_row(A, row, col, rows)
        
        if leading_row is None:
            continue
        
        if leading_row != row:
            A[[row, leading_row]] = A[[leading_row, row]]
        
        leading_element = A[row, col]
        A[row] /= leading_element
        
        for r in range(rows):
            if r != row:
                A[r] = abs(A[r] - A[row] * A[r, col])
        
        row += 1

    return A

A_RREF = RREF(A)
print("Базис RREF\n", A_RREF[np.any(A_RREF != 0, axis=1)])

# 1.3.1
S1 = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
             [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
S_REF = REF(S1)
print("S_REF:\n", S_REF)

# 1.3.2
G = S_REF[np.any(S_REF != 0, axis=1)]
k, n = G.shape
print(f'n = {n}, k = {k}')

# 1.3.3
# 1 Step
G_star = RREF(G)
print("G*:\n", G_star)
# 2 Step
row = 0
lead = []
# Фиксация ведущих столбцов
for col in range(n):
    for r in range(row, k):
        if G_star[r, col] != 0:
            lead.append(col)
            row += 1
            break
print(f'lead = {lead}')
# 3 Step
lead = np.array(lead)
mask = np.ones(G_star.shape[1], dtype=bool)
mask[lead] = False
X = G_star[:, mask]
print("X:\n", X)
# 4 Step
I = np.eye(np.shape(X)[1])
H = []
r_X = 0
r_I = 0

for col in range(len(X) + len(I)):
    if col in lead:
        if r_X < len(X):
            H.append(X[r_X])
            r_X += 1
    else:
        if r_I < len(I):
            H.append(I[r_I])
            r_I += 1

H = np.array(H)
print("H:\n", H)

# 1.4.1
def generate_code(S):
    code = set()
    code.add("0000")
    for word in S:
        code.add(word)

    # Сложение всех слов, оставление неповторяющихся + проверка размерности
    for i in range(len(S)):
        for j in range(len(S)):
            word = bin(int(S[i], 2) + int(S[j], 2))[2:].zfill(4)
            if len(word) > 4:
                word = word[-4:]
            code.add(word)

    return list(code)

PS = ["0100", "0011", "1100"]
C = generate_code(PS)
print("Код C =", C)

# 1.4.2
u = [1, 0, 1, 1, 0]
v = u@G
print("u@G = ", v%2)
print("v@H = ", (v@H)%2)

# 1.5
def d(G):
    d = len(G[0])
    # Используется broadcasting для сравнения всех пар кодовых слов одновременно
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            diff = np.sum(G[i] != G[j])
            d = min(d, diff)
    return d

d = d(G)
print("d =", d)
t = (d - 1) // 2
print("t =", t)

# 1.5.1 
e1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
print(f'\nv = {v}\nv+e1 = {(v+e1)%2}\n(v+e1)@H = {((v+e1)@H)%2}\n ..::ERROR::..')

# 1.5.2
def find_error_vector():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            error_vector = np.zeros_like(v, dtype=int)
            error_vector[i] = 1
            error_vector[j] = 1
            # Проверка, что вектор не образует ошибку
            ve2 = (v + error_vector) % 2
            ve2H = np.dot(ve2, H) % 2
            if np.all(ve2H == 0):
                return error_vector
    return None

e2 = find_error_vector()
print(f'\ne2 = {e2}\nv+e2 = {(v+e2)%2}\n(v+e2)@H = {((v+e2)@H)%2}\n ..::NO ERROR::..')
