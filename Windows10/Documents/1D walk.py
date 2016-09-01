# 2D walk problem by Ivan E. Perez 6:07pm 5/19/2016

from random import randint

def two_dimensional_randomwalk():
    steps = 0
    times = [0]*11
    for i in range(0,11):
        times[i] = [0]*11
    x, y = 5, 5
    moves = [(1,0), (0,1), (-1,0), (0,-1)]
    while x < 11 and x >= 0 or y < 11 and y >= 0:
        dx, dy = moves[randint(0,3)]
        x += dx
        y += dy
        if dx == 1 and dy == 0:
            print ("He moved right")
        elif dx == 0 and dy == 1:
            print ("He moved up")
        elif dx == -1 and dy == 0:
            print ("He moved left")
        elif dx == 0 and dy == -1:
            print ("He moved down")
        try:
            times[x][y] += 1
            steps += 1
        except IndexError:
            break

for i in range(0,11):
    for j in range(0,11):
        print ("He took {0} steps until he reached the end of the sidewalk.".format(steps), "He stood on {1}x{2} square at {0} times.".format(times[i][j], i+1,j+1)) 


# online code
def two_dimensional_random_walk():
    steps = 0
    times = [0] * 11
    for i in range(0,11):
        times[i] = [0] * 11
    x = 5
    y = 5
    moves = [(1,0), (0,1), (-1,0), (0,-1)]  
    while x<11 and x >= 0 or y < 11 and y >= 0:  
        dx, dy = moves[randint(0,3)]
        x += dx
        y += dy
        if dx == 1 and dy == 0:
            print("He moved right")
        elif dx == 0 and dy == 1:
            print("He moved up")
        elif dx == -1 and dy == 0:
            print("He moved left")
        elif dx == 0 and dy == -1:
            print("He moved down")
        try:
            times[x][y] += 1
            steps += 1
        except IndexError:
            break

for i in range(0,11):
    for j in range(0,11):
        print("He took {steps} steps until he reaches end of the sidewalk.".format(steps = steps),  "He stood on {1}x{2} square at {0} times".format(times[i][j], i+1,j+1) )  
