import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from .learning_config import LearningConfig, UpdationLevel
from .metric_factory import MetricHandlerContainer
from .progress_monitor import ProgressMonitor
from .training_context import TrainingContext
from lib.logging import wrap_in_logger
from lib.types import LearningMode


class Trainer:  # TODO: make it a singleton
    def __init__(
            self,
            net_factory_function_path: pathlib.PosixPath,
            learning_config: LearningConfig
    ):
        self._cntx = TrainingContext(
            net_factory_function_path,
            learning_config
        )
        self._progress_monitor = ProgressMonitor(
            sub_net_names=[
                x[0] for x in self._cntx.net.named_children()
            ],
            metrics=MetricHandlerContainer.from_config(
                learning_config.metrics
            ),
            tensorboard_writer=SummaryWriter(
                learning_config.tensorboard_logs
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
                "TrainingContext=..., "  # noqa: E131
                f"ProgressMonitor={self._progress_monitor}"  # noqa: E131
            ")"
        )

    @wrap_in_logger(level="debug")
    def run(self) -> None:  # TODO: return here after Dima's code review
        for epoch in range(self._cntx.total_epoch_amount):
            # with self._progress_monitor:  # TODO: re-think what epoch is; upd: it also causes ambuguity when expcetion is raised inside of the context manager  # noqa: E501
            self._progress_monitor._enter()
            self._process_dataset(LearningMode.TRAIN)
            self._process_dataset(LearningMode.VAL)
            self._progress_monitor._exit()
            self._progress_monitor.log_updation(
                UpdationLevel.EPOCH,
                LearningMode.TRAIN
            )
            self._progress_monitor.log_updation(
                UpdationLevel.EPOCH,
                LearningMode.VAL
            )
            if self._progress_monitor.best_moment.epoch == epoch:
                self._cntx.save_checkpoint("best")
            self._cntx.save_checkpoint("last")

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _process_dataset(self, mode: LearningMode) -> None:
        getattr(
            self._cntx.net,
            {LearningMode.TRAIN: "train", LearningMode.VAL: "eval"}[mode]
        )()
        for X, Y in self._cntx.dataloaders[mode]:
            if mode == LearningMode.TRAIN:
                self._cntx.optimizer.zero_grad()
            batch_loss = self._process_batch(X, Y, mode)
            if mode == LearningMode.TRAIN:
                self._cntx.optimizer.step()
                self._cntx.optimizer.zero_grad()
            self._cntx.lr_scheduler.step(UpdationLevel.GSTEP, batch_loss, mode)  # noqa: E501
            # TODO: may be next line should be moved after GA maintainance
            self._progress_monitor.log_updation(UpdationLevel.GSTEP, mode)
        self._cntx.lr_scheduler.step(
            UpdationLevel.EPOCH,
            self._progress_monitor.get_running_epoch_loss_value(mode),
            mode
        )

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _process_batch(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            /,
            mode: LearningMode
    ) -> float:
        X = X.to(self._cntx.device)
        Y = Y.to(self._cntx.device)
        Y_pred = self._cntx.net(X)
        loss_value = self._cntx.loss(Y_pred, Y)
        if mode == LearningMode.TRAIN:
            loss_value.backward()
        self._progress_monitor.record_batch_processing(
            mode, X, Y, Y_pred, loss_value, self._cntx
        )
        return loss_value.cpu().item()
