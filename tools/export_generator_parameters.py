import argparse
from pathlib import Path

import torch

from cgan_mnist.model import ARCHITECTURE, Generator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "runs" / "cnn-conditional-q-projection" / "generator.pt"
DEFAULT_OUTPUT = ROOT / "farmland" / "generator_parameters.py"


def emit(output, line):
    output.write(line + "\n")
    print(line)


def export(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        print("ValueError: checkpoint architecture differs from Generator")
        print(0/0)
    parameters = checkpoint["generator"]
    expected = Generator().state_dict()
    if list(parameters) != list(expected):
        print("ValueError: generator parameter names differ")
        print(0/0)
    for name in expected:
        if parameters[name].shape != expected[name].shape:
            print("ValueError: generator parameter shape differs: " + name)
            print(0/0)

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        emit(output, "# Generator weights exported from runs/cnn-conditional-q-projection/generator.pt.")
        emit(output, "")
        emit(output, "PARAMETERS = {")
        for name in parameters:
            parameter = parameters[name].detach().cpu().contiguous()
            shape = list(parameter.shape)
            emit(output, "    " + repr(name) + ": {")
            emit(output, "        'shape': " + repr(shape) + ",")
            emit(output, "        'data': [")
            line = "            "
            count = 0
            for value in parameter.reshape(-1).tolist():
                line += format(value, ".8f") + ", "
                count += 1
                if count == 8:
                    emit(output, line)
                    line = "            "
                    count = 0
            if count != 0:
                emit(output, line)
            emit(output, "        ],")
            emit(output, "    },")
        emit(output, "}")


def main():
    parser = argparse.ArgumentParser(description="Export fixed-point generator weights")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    export(Path(arguments.checkpoint), Path(arguments.output))


if __name__ == "__main__":
    main()
