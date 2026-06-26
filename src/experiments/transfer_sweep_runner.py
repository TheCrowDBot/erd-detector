from copy import deepcopy
from itertools import product

from src.schemas.sweep_result import SweepResult
from src.experiments.transfer_experiment_runner import TransferExperimentRunner


class TransferSweepRunner:

    def run(self, cfg, train_result, parameters, logger=None):

        results = []

        keys = list(parameters.keys())
        combinations = list(product(*parameters.values()))

        runner = TransferExperimentRunner()

        for i, values in enumerate(combinations, start=1):

            overrides = dict(zip(keys, values))

            print()
            print("=" * 60)
            print(f"Sweep {i}/{len(combinations)} testing {overrides}")
            print("=" * 60)

            current_cfg = deepcopy(cfg)

            current_cfg.transfer_model.overrides = {
                **(current_cfg.transfer_model.overrides or {}),
                **overrides,
            }

            result = runner.run(
                current_cfg,
                train_result=train_result,  # reused, NOT retrained
            )

            sweep_result = SweepResult(
                parameters=overrides,
                experiment_result=result,
            )

            results.append(sweep_result)

            if logger:
                logger.log(sweep_result)

        return results
