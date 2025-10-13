#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import sys

sys.path.append("src/libsvm")
sys.path.append("src/cv")
sys.path.append("src/fine_tuning/glue")
sys.path.append("src/fine_tuning/llm")
sys.path.append("src/fine_tuning/style")

from config import parse_args
from utils import get_run_name
from libsvm import main_libsvm
from cv import main_cv
from fine_tuning.glue import main_glue
from fine_tuning.llm import main_llm as _main_llm
from fine_tuning.style import main_style

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~~~~~ USING CPU ~~~~~~~~~~~~~~~")
    args, parser = parse_args()
    args.run_name = get_run_name(args, parser)

    if args.dataset.lower() in main_libsvm.DATASETS:
        main_libsvm.main(args, parser)
    elif args.dataset.lower() in main_cv.DATASETS:
        main_cv.main(args, parser)
    elif args.dataset.lower() in main_glue.DATASETS:
        main_glue.main(args)
    elif args.dataset.lower() in _main_llm.DATASETS:
        _main_llm.main(args)
    elif args.dataset.lower() == "style":
        main_style.main(args)
