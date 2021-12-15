"""
Entry point of project
"""

from pathlib import Path

import gdown

import ui


def google_drive_download(url, output) -> None:
    """Downloads file from google drive if output file does not exist."""
    if not Path(output).exists():
        gdown.download(url, output, quiet=False)


def main() -> None:
    # Download database
    google_drive_download(
        url='https://drive.google.com/uc?id=1ZC_qrjWnTTt61e628Ejh-bBLh6CRdnq_',
        output='carbonmonitor-us_datas_2021-11-04.csv'
    )

    # Download pre-trained model
    google_drive_download(
        url='https://drive.google.com/uc?id=1Hqgdfd8S2i-aBpdseV9h1ByjArJywEA3',
        output='model.keras'
    )

    # Download pre-trained model, trained with normalized data
    google_drive_download(
        url='https://drive.google.com/uc?id=1vodjV37FH3r7NfnMeYilteKwzYFZGPT8',
        output='normalized_data_model.keras'
    )

    ui.main()


if __name__ == "__main__":
    main()
