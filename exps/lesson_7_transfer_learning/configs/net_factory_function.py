def create_a_neural_network_instance():
    import pathlib
    import typing as tp
    from collections import OrderedDict
    from functools import reduce
    from itertools import chain

    import torch

    class Flatten(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(x.size(0), -1)

    class Bottleneck(torch.nn.Module):
        def __init__(
                self,
                conv1: torch.nn.Conv2d,
                bn1: torch.nn.BatchNorm2d,
                conv2: torch.nn.Conv2d,
                bn2: torch.nn.BatchNorm2d,
                conv3: torch.nn.Conv2d,
                bn3: torch.nn.BatchNorm2d,
                downsample: tp.Optional[torch.nn.Sequential]
        ):
            super().__init__()
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv1, self.conv2, self.conv3 = conv1, conv2, conv3
            self.bn1, self.bn2, self.bn3 = bn1, bn2, bn3
            self.downsample = downsample

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    def _bottleneck_factory(
            src_emb_size: int,
            middle_emb_size: int,
            tgt_emb_size: int,
            do_downsample: bool,
            /
    ) -> Bottleneck:
        return Bottleneck(
            conv1=torch.nn.Conv2d(
                in_channels=src_emb_size,
                out_channels=middle_emb_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            bn1=torch.nn.BatchNorm2d(
                num_features=middle_emb_size,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            conv2=torch.nn.Conv2d(
                in_channels=middle_emb_size,
                out_channels=middle_emb_size,
                kernel_size=(3, 3),
                stride=(2, 2) if do_downsample and (middle_emb_size > 64) else (1, 1),  # noqa: E501
                padding=(1, 1),
                bias=False
            ),
            bn2=torch.nn.BatchNorm2d(
                num_features=middle_emb_size,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            conv3=torch.nn.Conv2d(
                in_channels=middle_emb_size,
                out_channels=tgt_emb_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            bn3=torch.nn.BatchNorm2d(
                num_features=tgt_emb_size,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            downsample=None if not do_downsample else torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "0",
                            torch.nn.Conv2d(
                                in_channels=src_emb_size,
                                out_channels=tgt_emb_size,
                                groups=1,
                                kernel_size=(1, 1),
                                stride=(2, 2) if middle_emb_size > 64 else (1, 1),  # noqa: E501
                                padding=(0, 0),
                                bias=False,
                                padding_mode="zeros"
                            )
                        ),
                        (
                            "1",
                            torch.nn.BatchNorm2d(
                                num_features=tgt_emb_size,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True
                            )
                        )
                    ]
                )
            )
        )

    class ResNet50(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                groups=1,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
                padding_mode="zeros"
            )
            self.bn1 = torch.nn.BatchNorm2d(
                num_features=64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = torch.nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                return_indices=False,
                ceil_mode=False
            )
            self.layer1 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("0", _bottleneck_factory(64, 64, 256, True)),
                        ("1", _bottleneck_factory(256, 64, 256, False)),
                        ("2", _bottleneck_factory(256, 64, 256, False))
                    ]
                )
            )
            self.layer2 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("0", _bottleneck_factory(256, 128, 512, True)),
                        ("1", _bottleneck_factory(512, 128, 512, False)),
                        ("2", _bottleneck_factory(512, 128, 512, False)),
                        ("3", _bottleneck_factory(512, 128, 512, False))
                    ]
                )
            )
            self.layer3 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("0", _bottleneck_factory(512, 256, 1024, True)),
                        ("1", _bottleneck_factory(1024, 256, 1024, False)),
                        ("2", _bottleneck_factory(1024, 256, 1024, False)),
                        ("3", _bottleneck_factory(1024, 256, 1024, False)),
                        ("4", _bottleneck_factory(1024, 256, 1024, False)),
                        ("5", _bottleneck_factory(1024, 256, 1024, False))
                    ]
                )
            )
            self.layer4 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("0", _bottleneck_factory(1024, 512, 2048, True)),
                        ("1", _bottleneck_factory(2048, 512, 2048, False)),
                        ("2", _bottleneck_factory(2048, 512, 2048, False))
                    ]
                )
            )
            self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
            self.flatten = Flatten()
            self.fc = torch.nn.Linear(
                in_features=2048,
                out_features=1000,
                bias=True
            )
            self._modules_order = [
                "conv1", "bn1", "relu", "maxpool",
                "layer1", "layer2", "layer3", "layer4",
                "avgpool", "flatten",
                "fc"
            ]

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return reduce(
                lambda x, f: f(x),
                (getattr(self, name) for name in self._modules_order),
                input_tensor
            )

        def remove_layer(self, layer_name: str) -> None:
            self._modules_order.remove(layer_name)
            del self._modules[layer_name]

    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = self._produce_the_backbone()
            self.head = torch.nn.Linear(
                in_features=2048,
                out_features=50,
                bias=True
            )

        @staticmethod
        def _produce_the_backbone() -> torch.nn.Module:
            future_backbone = ResNet50()
            _weights_dir = pathlib.Path("/var/lib/storage/resources/weights")
            future_backbone.load_state_dict(
                torch.load(
                    _weights_dir / "resnet50_weights.imagenet1k_v1.stt"
                )["model_state_dict"]
            )
            for param in chain(
                    future_backbone.conv1.parameters(),
                    future_backbone.bn1.parameters(),
                    future_backbone.relu.parameters(),
                    future_backbone.maxpool.parameters(),
                    future_backbone.layer1.parameters(),
                    future_backbone.layer2.parameters(),
                    future_backbone.layer3.parameters()
            ):
                param.requires_grad = False
            future_backbone.remove_layer("fc")
            return future_backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.backbone(x)
            return self.head(out)
    return NeuralNetwork()
