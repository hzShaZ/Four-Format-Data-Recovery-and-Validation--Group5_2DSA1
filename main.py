"""Entry point: python main.py --config config.json"""
import argparse
import json
import sys


def main() -> int:
    """Run the pipeline; return a nonzero exit status on failure (member 5)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    print(f"Loaded config with {len(config['inputs'])} inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())