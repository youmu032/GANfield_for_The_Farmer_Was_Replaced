from config import M, N
def move_to(tx, ty):
    x, y=get_pos_x(), get_pos_y()
    if tx>x:
        for i in range(tx-x):
            move(East)
    elif tx<x:
        for i in range(x-tx):
            move(West)

    if ty>y:
        for i in range(ty-y):
            move(North)
    elif ty<y:
        for i in range(y-ty):
            move(South)

def dist(p1, p2):
    x,y=p1
    tx,ty=p2
    return abs(tx-x)+abs(ty-y)
