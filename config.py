data_path = "/data/"
# data_path = "./"


def get_train_valid_paths(dir_name):
    return f"{data_path}{dir_name}/train", f"{data_path}{dir_name}/validation"

