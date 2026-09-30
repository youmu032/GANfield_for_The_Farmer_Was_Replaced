from tensor import tensor, zeros


def linear(inputs, weight, bias=None):
    batch, input_size = inputs["shape"]
    output_size, weight_input_size = weight["shape"]
    if input_size != weight_input_size or (bias != None and bias["shape"] != [output_size]):
        print("ValueError: linear dimensions differ")
        print(0/0)
    result = []
    for index in range(batch * output_size):
        result.append(0.0)
    for n in range(batch):
        for output in range(output_size):
            if bias == None:
                total = 0.0
            else:
                total = bias["data"][output]
            for index in range(input_size):
                total += inputs["data"][n * input_size + index] * weight["data"][output * input_size + index]
            result[n * output_size + output] = total
    return tensor([batch, output_size], result)


def relu(inputs):
    output = []
    for value in inputs["data"]:
        if value > 0:
            output.append(value)
        else:
            output.append(0.0)
    return tensor(inputs["shape"], output)


def conv_transpose2d(inputs, weight, bias=None, stride=2, padding=1):
    batch, input_channels, height, width = inputs["shape"]
    weight_channels, output_channels, kernel_h, kernel_w = weight["shape"]
    if input_channels != weight_channels or (bias != None and bias["shape"] != [output_channels]):
        print("ValueError: convolution dimensions differ")
        print(0/0)
    output_h = (height - 1) * stride - 2 * padding + kernel_h
    output_w = (width - 1) * stride - 2 * padding + kernel_w
    result = zeros([batch, output_channels, output_h, output_w])
    output_data = result["data"]
    input_data = inputs["data"]
    weights = weight["data"]
    for n in range(batch):
        for channel in range(input_channels):
            for row in range(height):
                for col in range(width):
                    source = input_data[((n * input_channels + channel) * height + row) * width + col]
                    for out_channel in range(output_channels):
                        kernel_offset = (channel * output_channels + out_channel) * kernel_h * kernel_w
                        for kh in range(kernel_h):
                            out_row = row * stride - padding + kh
                            if out_row < 0 or out_row >= output_h:
                                continue
                            for kw in range(kernel_w):
                                out_col = col * stride - padding + kw
                                if 0 <= out_col < output_w:
                                    target = ((n * output_channels + out_channel) * output_h + out_row) * output_w + out_col
                                    output_data[target] += source * weights[kernel_offset + kh * kernel_w + kw]
    if bias != None:
        plane = output_h * output_w
        for n in range(batch):
            for channel in range(output_channels):
                offset = (n * output_channels + channel) * plane
                for index in range(plane):
                    output_data[offset + index] += bias["data"][channel]
    return result


def instance_norm2d(inputs, weight=None, bias=None, eps=0.00001):
    batch, channels, height, width = inputs["shape"]
    if weight != None and weight["shape"] != [channels]:
        print("ValueError: normalization weight dimensions differ")
        print(0/0)
    if bias != None and bias["shape"] != [channels]:
        print("ValueError: normalization bias dimensions differ")
        print(0/0)
    plane = height * width
    output = []
    for index in range(len(inputs["data"])):
        output.append(0.0)
    for n in range(batch):
        for channel in range(channels):
            offset = (n * channels + channel) * plane
            mean = 0.0
            for index in range(plane):
                mean += inputs["data"][offset + index]
            mean /= plane
            variance = 0.0
            for index in range(plane):
                difference = inputs["data"][offset + index] - mean
                variance += difference * difference
            variance /= plane
            if weight == None:
                scale = 1.0
            else:
                scale = weight["data"][channel]
            if bias == None:
                shift = 0.0
            else:
                shift = bias["data"][channel]
            divisor = (variance + eps) ** 0.5
            for index in range(plane):
                output[offset + index] = (inputs["data"][offset + index] - mean) / divisor * scale + shift
    return tensor(inputs["shape"], output)


def tanh(inputs):
    output = []
    for value in inputs["data"]:
        if value >= 20:
            output.append(1.0)
        elif value <= -20:
            output.append(-1.0)
        else:
            exponent = 2.718281828459045 ** (2 * value)
            output.append((exponent - 1) / (exponent + 1))
    return tensor(inputs["shape"], output)
