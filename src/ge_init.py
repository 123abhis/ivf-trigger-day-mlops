import great_expectations as gx
from pathlib import Path

def init_ge():
    ge_root = Path("great_expectations")

    context = gx.get_context(
        context_root_dir=ge_root
    )

    print("Great Expectations initialized at:", ge_root.resolve())

if __name__ == "__main__":
    init_ge()
