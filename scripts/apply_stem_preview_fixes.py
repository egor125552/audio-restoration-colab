"""One-off migration marker for the stem preview fix.

The source changes have already been applied to the branch. This small module is
kept only so old manually dispatched verification workflows do not reference a
missing file.
"""


def main() -> None:
    return None


if __name__ == "__main__":
    main()
