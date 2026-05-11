import time
from pathlib import Path


def wait_until_file_complete(file_path, check_interval=1):

    file_path = Path(file_path)

    previous_size = -1

    while True:

        if not file_path.exists():
            return False

        current_size = file_path.stat().st_size

        if current_size == previous_size:
            return True

        previous_size = current_size

        time.sleep(check_interval)