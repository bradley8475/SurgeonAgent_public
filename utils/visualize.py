# visualize the json from a directory

# generate a static html page that shows the log data

import json

# css styling
css_styling = """

"""


def load_log_data(log_path: str):
    # read the log file
    with open(log_path, "r") as f:
        log_data = json.load(f)
    return log_data


def visualize_logs(log_path: str):
    log_data = load_log_data(log_path)

    # visualize the log data


if __name__ == "__main__":
    visualize_logs("logs/20250827_115905/manager/20250827_115905_response.json")
