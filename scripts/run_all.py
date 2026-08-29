import sys

from vip_eeg.cli import main

if __name__ == "__main__":
    main(["run-all", *sys.argv[1:]])
