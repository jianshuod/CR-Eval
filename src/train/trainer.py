import os
import torch
import wandb
import datasets
import torch.nn as nn
from typing import Union
from typing import Optional
from src.utils import print_rank_0
from torch.utils.data import Sampler
from src.train.data.sampler import StageSampler
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import seed_worker
from transformers import Seq2SeqTrainer as HFSeq2SeqTrainer
from src.train.utils import generate_model_card, get_comet_experiment_url
from transformers import Trainer, is_datasets_available, is_wandb_available
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)

PREFIX_CHECKPOINT_DIR = "checkpoint"


class CustomDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        # Call the sampler's set_epoch if it exists
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

        # Also call the dataset's set_epoch if it exists
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class CRiticTrainer(HFSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):

        self.curriculum_learning = kwargs.pop("curriculum_learning", False)
        self.eval_data_collator = kwargs.pop("eval_data_collator", None)

        super(CRiticTrainer, self).__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.curriculum_learning:
            task_order = [
                "general",
                "outline-revision",
                "diff-analysis",
                "explain-vuln",
                "fill-cr",
            ]
            task_names = getattr(self.train_dataset, "task_names", None)
            if task_names is None:
                print_rank_0(
                    "WARNING: task names not found in dataset, using default RandomSampler"
                )
            else:
                print_rank_0("Curriculum learning has been activated.")
                stage_indices = [
                    [
                        i
                        for i, task_name in enumerate(task_names)
                        if task_name == target_task
                    ]
                    for target_task in task_order
                ]
                return StageSampler(stage_indices)
        elif len(self.args.learn_step_by_step) != 1:  # default to be ["others"]
            task_order = self.args.learn_step_by_step
            # determine the types included in the others type
            all_candidate_task_names = set(task_order).remove("others")

            task_names = getattr(self.train_dataset, "task_names", None)
            if task_names is None:
                print_rank_0(
                    "WARNING: task names not found in dataset, using default RandomSampler"
                )
            else:
                stage_indices = [
                    [
                        i
                        for i, task_name in enumerate(task_names)
                        if (target_task != "others" and task_name == target_task)
                        or (
                            target_task == "others"
                            and task_name not in all_candidate_task_names
                        )
                    ]
                    for target_task in task_order
                ]
                return StageSampler(stage_indices)

        return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        print_rank_0(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        print_rank_0(f"bitsandbytes: will optimize {module} in fp32")
                print_rank_0(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(CRiticTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(CRiticTrainer, self)._save(output_dir, state_dict)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            CustomDataLoader(train_dataset, **dataloader_params)
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )

        #! Custom data_collator for evaluation
        if hasattr(self, "eval_data_collator"):
            data_collator = self.eval_data_collator
        else:
            data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=(
                wandb.run.get_url()
                if is_wandb_available() and wandb.run is not None
                else None
            ),
            comet_url=get_comet_experiment_url(),
            trainer_name="Iterative SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


from transformers import EarlyStoppingCallback

from transformers.utils import logging


logger = logging.get_logger(__name__)


class EarlyStoppingAfterOneEpochCallback(EarlyStoppingCallback):

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if (
            state.epoch >= 1
            and self.early_stopping_patience_counter >= self.early_stopping_patience
        ):
            control.should_training_stop = True


from transformers import TrainerCallback


class DataAugAfterOneEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Extract the trainer from kwargs to access the dataset
        trainer = kwargs.get("trainer")

        # Check if the train dataset has the `_dataset_evolving` method
        if trainer and hasattr(trainer.train_dataset, "_dataset_evolving"):
            # Call the data augmentation function within the dataset
            trainer.train_dataset._dataset_evolving()
            print_rank_0("Managed to evolve.")

        # Ensure the super method is called for potential chaining
        return super().on_epoch_end(args, state, control)


class CRiticDatasetCommunicationCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):

        train_dataset = kwargs.get("train_dataloader").dataset
        model = kwargs.get("model")
        data_collator = kwargs.get("train_dataloader").collate_fn

        if hasattr(train_dataset, "communicate_with_trainer"):
            train_dataset.communicate_with_trainer(
                state.epoch, start=0, model=model, args=args, collate_fn=data_collator
            )

    def on_epoch_begin(self, args, state, control, **kwargs):

        train_dataset = kwargs.get("train_dataloader").dataset
        model = kwargs.get("model")
        data_collator = kwargs.get("train_dataloader").collate_fn

        if hasattr(train_dataset, "communicate_with_trainer"):
            train_dataset.communicate_with_trainer(
                state.epoch, start=1, model=model, args=args, collate_fn=data_collator
            )

    def on_train_end(self, args, state, control, **kwargs):

        train_dataset = kwargs.get("train_dataloader").dataset

        if hasattr(train_dataset, "on_train_end"):
            train_dataset.on_train_end()
