from generator import make_generator
from tensor import tensor
from plant_img import plant_img

def noise_vector():
    # global random_state
    # random_state = 123456789
    values = []
    for index in range(32):
        value = 0.0
        for sample in range(12):
            value += random()
        values.append(value - 6.0)
    return tensor([1, 32], values)


def main(number):
    if number < 0 or number > 9:
        print("ValueError: digit must be between 0 and 9")
        print(0/0)
    generated = make_generator()(noise_vector(), [number])
    binary_image = []
    for row in range(16):
        binary_row = []
        text_row = ""
        for column in range(16):
            value = generated["data"][row * 16 + column]
            if value >= 0.0:
                binary_row.append(1)
                text_row += "1"
            else:
                binary_row.append(0)
                text_row += "0"
        binary_image.append(binary_row)
        quick_print(text_row)
    return binary_image

tlist=[1,1,4,5,1,4,1,9,1,9,8,1,0]
for i in tlist:
    bin_image=main(i)
    plant_img(bin_image)

while True:
    do_a_flip()
