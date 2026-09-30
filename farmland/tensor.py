def size(shape):
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def tensor(shape, data):
    if size(shape) != len(data):
        print("ValueError: shape does not match data length")
        print(0/0)
    return {"shape": list(shape), "data": list(data)}


def zeros(shape):
    data = []
    for index in range(size(shape)):
        data.append(0.0)
    return tensor(shape, data)


def reshape(value, shape):
    return tensor(shape, value["data"])


def from_nested(values):
    rows = len(values)
    if rows == 0:
        return tensor([0, 0], [])
    columns = len(values[0])
    data = []

    for row in values:
        if len(row) != columns:
            print("ValueError: nested list must be rectangular")
            print(0/0)
        for value in row:
            data.append(value)
    return tensor([rows, columns], data)


def to_nested(value):
    shape = value["shape"]
    data = value["data"]

    def build(depth, offset):
        if depth == len(shape):
            return data[offset]
        step = size(shape[depth + 1:])
        result = []
        for index in range(shape[depth]):
            result.append(build(depth + 1, offset + index * step))
        return result

    return build(0, 0)


def concat_columns(left, right):
    if len(left["shape"]) != 2 or len(right["shape"]) != 2:
        print("ValueError: expected two matrices")
        print(0/0)
    rows, left_width = left["shape"]
    right_rows, right_width = right["shape"]
    if rows != right_rows:
        print("ValueError: batch sizes differ")
        print(0/0)
    output = []
    for row in range(rows):
        output += left["data"][row * left_width:(row + 1) * left_width]
        output += right["data"][row * right_width:(row + 1) * right_width]
    return tensor([rows, left_width + right_width], output)
