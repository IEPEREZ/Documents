#updated 2D random walk
# this update uses trivializes the repeptive if/elif structure to a while loop

def two_dimensional_random_walk():
    steps = 0
    grid_size = 11
    times = [0]*grid_size
    for i in range(0,grid_size):
        times[i] = [0]*grid_size
    x, y = 5, 5
    moves = [(1, 0, "right"), (0,1, "up"), (-1,0, "left"), (0, -1, "down")]
    while True:
        dx, dy, position = moves[randint(0,3)]
        x += dx
        y += dy
        print ("He moved", position)
        try:
            times[x][y] += 1
            steps += 1
        except IndexError:
            break
    for i in range(0,11):
        for j in range(0,11):
            print("He took {0} steps until he reached the end of the sidewalk.".format(steps),  "He stood in {1}x{2} squares at {0} times".format(times[i][j], i+1, j+1))
            
print two_dimensional_random_walk()
