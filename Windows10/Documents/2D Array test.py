#2D Array by Ivan E Perez
size = 100
Matrix = [[0 for i in range(size)]for j in range(size)]

Matrix[50][50] = 2
Matrix[25][25] = 1

print Matrix[50][50]
i, j = 25, 25
print Matrix[i][j]

times = [0]*11
for i in range(0,11):
    times[i] = [0]*11
print times[i]
