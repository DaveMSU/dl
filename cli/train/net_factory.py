import importlib.util
import pathlib
import sys

import torch


class NetFactory:
    @staticmethod
    def create_network(function_path: pathlib.PosixPath) -> torch.nn.Module:
        spec = importlib.util.spec_from_file_location(
            "_net_factory_function_impl",
            function_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["_net_factory_function_impl"] = module
        return getattr(module, "create_a_neural_network_instance")()
