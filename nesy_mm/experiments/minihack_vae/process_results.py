import wandb
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    # TODO: Implement also for offline wandb folder
    parser = ArgumentParser()
    parser.add_argument("--wandb_org", type=str)
    parser.add_argument("--wandb_project", type=str)
    args = parser.parse_args()

    api = wandb.Api()
    entity, project = args.wandb_org, args.wandb_project
    runs = api.runs(entity + "/" + project)

    data = []
    time = []
    for run in runs:
        if run.name == "minihack-vae":
            print(f"Extracting info from run {run.id}...")

            # extract relevant data
            grid_size = run.config["grid_size"]
            model = run.config["model"]
            test_reconstruction_error = run.summary["test_reconstruction_error"]
            test_generative_accuracy = run.summary["test_generative_accuracy"]
            training_time = run.summary["train_time"]
            n_epochs = run.config["epochs"]

            # append to list
            data.append(
                {
                    "grid_size": grid_size,
                    "model": model,
                    "test_reconstruction_error": test_reconstruction_error,
                    "test_generative_accuracy": test_generative_accuracy,
                }
            )
            time.append(
                {
                    "grid_size": grid_size,
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
    df = pd.DataFrame(data)
    df_time = pd.DataFrame(time)

    # print(df)
    # print(df_time)

    # compute mean and standard error per grid_size per model
    grouped = df.groupby(["grid_size", "model"]).agg(
        {
            "test_reconstruction_error": ["mean", "sem"],
            "test_generative_accuracy": ["mean", "sem"],
        }
    )
    time_grouped = df_time.groupby(["grid_size", "model"]).agg(
        {"time": ["mean", "sem"]}
    )

    # multiply reconstruction error by 1000
    grouped["test_reconstruction_error"] *= 1000

    # multiply generative accuracy by 100
    grouped["test_generative_accuracy"] *= 100

    print(grouped)
    print(time_grouped)
