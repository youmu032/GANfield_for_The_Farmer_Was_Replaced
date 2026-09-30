from tensor import concat_columns, reshape, tensor
from ops import linear, relu, conv_transpose2d, instance_norm2d, tanh


def embedding(labels, weight):
    classes, width = weight["shape"]
    values = []
    for label in labels:
        if label < 0 or label >= classes:
            print("ValueError: label outside embedding table")
            print(0/0)
        values += weight["data"][label * width:(label + 1) * width]
    return tensor([len(labels), width], values)


def make_generator(parameters=None):
    if parameters == None:
        from generator_parameters import PARAMETERS
        parameters = PARAMETERS
    def forward(noise, labels):
        if noise["shape"] != [len(labels), 32]:
            print("ValueError: noise must have shape [batch, 32]")
            print(0/0)
        condition = embedding(labels, parameters["label_embedding.weight"])
        features = concat_columns(noise, condition)
        features = relu(linear(features, parameters["project.0.weight"], parameters["project.0.bias"]))
        features = reshape(features, [len(labels), 16, 4, 4])
        features = conv_transpose2d(features, parameters["upsample.0.weight"], parameters["upsample.0.bias"])
        features = instance_norm2d(features, parameters["upsample.1.weight"], parameters["upsample.1.bias"])
        features = relu(features)
        features = conv_transpose2d(features, parameters["upsample.3.weight"], parameters["upsample.3.bias"])
        return tanh(features)

    return forward
