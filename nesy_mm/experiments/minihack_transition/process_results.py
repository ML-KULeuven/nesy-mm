import os
import yaml
import pandas as pd
from argparse import ArgumentParser


def get_training_times(org, project):
    import wandb

    api = wandb.Api()
    entity, project = org, project
    runs = api.runs(entity + "/" + project)

    time = []
    for run in runs:
        if run.name == "minihack-transition":
            print(f"Extracting info from run {run.id}...")

            # extract relevant data
            model = run.config["model"]
            training_time = run.summary["train_time"]
            n_epochs = run.config["epochs"]

            time.append(
                {
                    "model": model,
                    "time": training_time * n_epochs,
                }
            )

    # Set display options to show all rows and columns
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option(
        "display.colheader_justify", "left"
    )  # Align column headers to the left

    # convert list to dataframe
    df_time = pd.DataFrame(time)

    print(df_time)

    # compute mean and standard error per grid_size per model
    time_grouped = df_time.groupby(["model"]).agg({"time": ["mean", "sem"]})

    print(time_grouped)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--res_dir", type=str, default="")
    parser.add_argument("--wandb_org", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    args = parser.parse_args()

    models = ["nesymm", "deephmm", "transformer"]

    configurations = []
    configurations += [(10, 1, 10)]
    configurations += [(10, 2, 10)]
    configurations += [(20, 1, 10)]
    configurations += [(20, 2, 10)]
    configurations += [(10, 1, 15)]
    configurations += [(10, 2, 15)]
    configurations += [(20, 1, 15)]
    configurations += [(20, 2, 15)]

    data = []
    for configuration in configurations:
        for model in models:
            for seed in range(5):
                model_dir = (
                    f"{model}_minihack-transition_10x10_horizon10_enemies1_seed{seed}"
                )
                yaml_file = os.path.join(
                    args.res_dir,
                    model_dir,
                    f"grid{configuration[2]}_horizon{configuration[0]}_enemies{configuration[1]}.yaml",
                )

                if os.path.exists(yaml_file):
                    with open(yaml_file, "r") as file:
                        res = yaml.load(file, Loader=yaml.FullLoader)

                        test_balanced_accuracy = None
                        test_f1_score = None
                        for r in res:
                            if "test_balanced_accuracy" == r[0]:
                                test_balanced_accuracy = r[1]
                            if "test_f1_score" == r[0]:
                                test_f1_score = r[1]

                        assert test_balanced_accuracy is not None
                        assert test_f1_score is not None
                        data.append(
                            {
                                "model": model,
                                "grid_size": configuration[2],
                                "horizon": configuration[0],
                                "n_enemies": configuration[1],
                                "test_balanced_accuracy": test_balanced_accuracy,
                                "test_f1_score": test_f1_score,
                            }
                        )
                else:
                    print(f"File not found: {yaml_file}")

    # Set display options to show all rows and columns
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option(
        "display.colheader_justify", "left"
    )  # Align column headers to the left
    pd.set_option("display.expand_frame_repr", False)  # Prevent wrapping to new line

    df = pd.DataFrame(data)
    # print(df)

    # compute mean and std of balanced accuracy and f1 per model per grid_size per horizon per n_enemies
    grouped = df.groupby(["model", "grid_size", "horizon", "n_enemies"]).agg(
        {"test_balanced_accuracy": ["mean", "sem"], "test_f1_score": ["mean", "sem"]}
    )
    grouped["test_balanced_accuracy"] *= 100
    print(grouped)

    get_training_times(args.wandb_org, args.wandb_project)
