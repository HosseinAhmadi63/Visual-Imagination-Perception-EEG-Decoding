import sys

from vip_eeg.cli import main

if __name__ == "__main__":
    main(["verify", *sys.argv[1:]])
