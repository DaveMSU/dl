def create_a_neural_network_instance():
    from collections import OrderedDict

    import torch


    class Flatten(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(x.size(0), -1)


    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv2d_0",
                            torch.nn.Conv2d(
                                in_channels=3,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=1,
                                padding=0
                            )
                        ),
                        (
                            "relu_1",
                            torch.nn.ReLU()
                        ),
                        (
                            "maxpool2d_2",
                            torch.nn.MaxPool2d(
                                kernel_size=(2, 2),
                                stride=2,
                                padding=0
                            )
                        ),
                        (
                            "flatten_3",
                            Flatten()
                        )
                    ]
                )
            )
            self.head = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear_4",
                            torch.nn.Linear(
                                in_features=10816,
                                out_features=128
                            )
                        ),
                        (
                            "relu_5",
                            torch.nn.ReLU()
                        ),
                        (
                            "linear_6",
                            torch.nn.Linear(
                                in_features=128,
                                out_features=10
                            )
                        )
                    ]
                )
            )
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))
    return NeuralNetwork()
