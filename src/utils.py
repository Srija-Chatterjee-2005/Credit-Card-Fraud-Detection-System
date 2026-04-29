import os


def create_project_folders():
    folders = [
        "models",
        "outputs",
        "outputs/charts",
        "outputs/reports",
        "outputs/logs",
        "images"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)