import torch


# TODO: re-design it, to allow easy re-weignting
class CrossEntropyLossWithDimensionReducingFirst(torch.nn.CrossEntropyLoss):
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        input, target = input, target.long()  # TODO: horrible crutch, implement a transform
        target = target.masked_fill(target == -1, 905)
        # TODO: probably it will fail somewhen and .contiguous() is the remedy
        return super().forward(
            input.view(-1, input.shape[-1]),
            target.view(-1)
        )
