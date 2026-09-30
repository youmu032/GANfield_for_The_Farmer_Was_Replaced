from tools import move_to
from seq_list import snake_seq
from config import M, N, wather_threhold

clear()

def check_plant(x, y, img):
    if img[N-y-1][x]==0:
        return Entities.Carrot
    else:
        return Entities.Tree

def init():
    for i, j in snake_seq(M,N):
        move_to(i, j)
        plant_type=check_plant(i, j)
        if plant_type==Entities.Carrot:
            till()

def get_till_requried_list():
    return [Entities.Carrot, Entities.Sunflower, Entities.Cactus]

def my_plant(x, y, img):
    now_plant=check_plant(x, y, img)

    if can_harvest() or get_entity_type()==None or get_entity_type()==Entities.Grass:
        if can_harvest():
            harvest()
        if now_plant in get_till_requried_list() and get_ground_type()!= Grounds.Soil:
            till()
        if now_plant not in get_till_requried_list() and get_ground_type()== Grounds.Soil:
            till()
        plant(now_plant)
        while get_water()<wather_threhold:
            use_item(Items.Water)
# init()
def plant_img(img):
    seq_list=snake_seq(M,N)
    for i, j in seq_list:
        move_to(i, j)
        my_plant(i, j, img)

def main():
    img=[]
    for i in range(M):
        img.append([])
        for j in range(N):
            t=0
            if j<=3:
                t=1
            img[-1].append(t)

    plant_img(img)
    while True:
        do_a_flip()