import numpy as np
import argparse, os


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--pred_scores_file",
    type=str,
    required=True,
    help="the scores predicted by our lovely detector",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
)
parser.add_argument("--cutoff_quantile", type=float, required=True, default=0.9)


def main(args):
    with open(args.pred_scores_file, "rb") as f:
        results = np.load(f, allow_pickle=True)  # a numpy array, not a dict
        results = results[()]  # a dict
    f.close()

    # sort images
    images_sorted_ascend = sorted(results.items(), key=lambda x: x[1])
    cutoff_point = int(args.cutoff_quantile * len(images_sorted_ascend))
    images_sorted_ascend = images_sorted_ascend[:cutoff_point]

    # write to
    file_name = args.pred_scores_file.split(".")[0]
    file_name = file_name.replace(
        "pred_scores_", "cutoff_" + str(args.cutoff_quantile) + "_"
    )

    with open(os.path.join(args.output_folder, file_name + ".txt"), "w") as f:
        f.writelines([image_path + "\n" for image_path, score in images_sorted_ascend])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
