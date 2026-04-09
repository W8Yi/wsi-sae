from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable


COMMAND_MODULES = {
    "train": "wsi_sae.commands.train",
    "mine": "wsi_sae.commands.mine",
    "build-prototypes": "wsi_sae.commands.build_prototypes",
    "build-targets": "wsi_sae.commands.build_targets",
    "compute-percentiles": "wsi_sae.commands.compute_percentiles",
    "probe": "wsi_sae.commands.probe",
    "export-viewer": "wsi_sae.commands.export_viewer",
}


def _usage() -> str:
    names = ", ".join(sorted(COMMAND_MODULES.keys()))
    return (
        "Usage: wsi-sae <command> [args...]\n\n"
        f"Commands: {names}\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage(), file=sys.stderr if args else sys.stdout)
        return 0 if args else 1

    command = args[0]
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        return 2

    module = import_module(module_name)
    entry: Callable[[], object] = getattr(module, "main")
    sys.argv = [f"wsi-sae {command}", *args[1:]]
    entry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

