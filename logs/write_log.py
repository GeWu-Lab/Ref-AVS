import os
from datetime import datetime

DEFAULT = "/home/yaoting_wang/workplace/Mask2Former/AVS/logs/log"

def write_log(message, _file_name, _dir_name=DEFAULT, tag=None, once=False):
    file_path = f"{_dir_name}/{_file_name}"

    with open(file_path, 'a') as file:
        if once:
            file.write(f'>>> {"="*60}\n')
            currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f\n")
            file.write(f"--- {currentDateAndTime}\n")
            file.write(f"--- Tag: {tag}\n")
        file.write(f'--- {message}\n')
        file.close()

