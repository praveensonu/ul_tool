from pathlib import Path

def print_tree(directory, prefix=""):
    path = Path(directory)
    items = sorted(path.iterdir())

    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(item, prefix + extension)

print_tree("/raid/p.bushipaka/ascent_2/")