size = 100
posn = [[[0, 0, 0, 0] for i in range(0, size)] for j in range(0, size) for k in range(0, size)]
newposn = [[[0, 0, 0, 0] for i in range(0, size)] for j in range(0, size) for k in range(0, size)]

posn[50][50] = [0.5, 0.5*j, 0.5, 0.5*i]

left, right = posn[j][k][0], posn[j][k][2]
up, down = posn[j][k][1], posn[j][k][3]

a = 0.5
newposn[j+1][k][0] = (a)*((-left)+up+right+down)
newposn[j-1][k][1] = (a)*(left-up+right+down)

newposn[j][k+1][2] = (a)*(left+up-right+down)
newposn[j][k-1][3] = (a)*(left+up+right-down)

def p(complex_number):
    return complex_number ** 2

probs = [[0 for i in range(len(posn[j]))] for j in range(len(posn))]
for i in range(len(posn)):
    for j in range(len(posn([i]))):
                   probs[i][j] = sum([p(a) for a in posn[i][j]])

print (probs[i][j])
#analysis methods: shap dist. new features of coin#
# Grovers Algorithm:
# used to to search unsorted DB, classically cannot be performed than linear time
                   # grover performs searchin O(logN)space and O(N ** (1/2))time.


