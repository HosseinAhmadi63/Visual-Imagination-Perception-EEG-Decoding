import sys

from vip_eeg.cli import main

if __name__ == "__main__":
    main(["figures", *sys.argv[1:]])
