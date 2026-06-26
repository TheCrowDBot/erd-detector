from src.schemas.sweep_result import SweepResult


class Leaderboard:

    def print(
        self,
        results: list[SweepResult],
    ) -> None:

        print()

        print("| parameter | value | map50-95 |")
        print("|------------|-------|----------|")

        for result in sorted(
            results,
            key=lambda x: x.experiment_result.evaluation_result.map5095,
            reverse=True,
        ):

            print(
                f"| {result.parameter} "
                f"| {result.value} "
                f"| {result.experiment_result.evaluation_result.map5095:.4f} |"
            )
