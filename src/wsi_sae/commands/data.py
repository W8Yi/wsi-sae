from __future__ import annotations

import argparse
import json

from wsi_sae.data.layout import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT,
    build_registry,
    ingest_tcga_features,
    init_layout,
    parse_encoder_list,
    promote_links,
    scan_h5_health,
    validate_layout,
)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Manage canonical WSI/feature data layout for wsi-sae.")
    sub = ap.add_subparsers(dest="data_command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=str, default=DEFAULT_DATA_ROOT, help="Canonical data root.")
    common.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Project name, e.g. TCGA.")
    common.add_argument("--encoders", type=str, default="uni2,seal", help="Comma-separated encoders.")

    p_init = sub.add_parser("init-layout", parents=[common], help="Create the canonical directory layout.")
    p_init.set_defaults(handler=_cmd_init_layout)

    p_ingest = sub.add_parser("ingest-tcga-features", parents=[common], help="Link legacy TCGA feature files into the canonical layout.")
    p_ingest.add_argument("--legacy-root", type=str, required=True, help="Legacy cohort-first TCGA feature root.")
    p_ingest.add_argument("--link-mode", type=str, default="symlink", choices=["symlink"], help="How to materialize canonical feature paths.")
    p_ingest.set_defaults(handler=_cmd_ingest)

    p_registry = sub.add_parser("build-registry", parents=[common], help="Build registry CSVs from the canonical layout.")
    p_registry.set_defaults(handler=_cmd_build_registry)

    p_validate = sub.add_parser("validate-layout", parents=[common], help="Validate canonical layout integrity.")
    p_validate.set_defaults(handler=_cmd_validate_layout)

    p_promote = sub.add_parser("promote-links", parents=[common], help="Replace canonical symlinks with real moved files.")
    p_promote.set_defaults(handler=_cmd_promote_links)

    p_scan = sub.add_parser("scan-h5-health", parents=[common], help="Scan H5 files for unreadable/corrupted feature payloads.")
    p_scan.add_argument("--source", type=str, default="canonical", choices=["canonical", "legacy"], help="Which tree to scan.")
    p_scan.add_argument("--legacy-root", type=str, default=None, help="Legacy TCGA feature root when --source=legacy.")
    p_scan.add_argument("--out-dir", type=str, default=None, help="Optional directory for CSV/JSON scan reports.")
    p_scan.add_argument("--stop-on-error", action="store_true", help="Raise immediately on first unreadable file.")
    p_scan.set_defaults(handler=_cmd_scan_h5_health)

    return ap


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_init_layout(args: argparse.Namespace) -> None:
    payload = init_layout(args.root, project=args.project, encoders=parse_encoder_list(args.encoders))
    _print_payload(payload)


def _cmd_ingest(args: argparse.Namespace) -> None:
    payload = ingest_tcga_features(
        args.root,
        legacy_root=args.legacy_root,
        project=args.project,
        encoders=parse_encoder_list(args.encoders),
        link_mode=args.link_mode,
    )
    _print_payload(payload)


def _cmd_build_registry(args: argparse.Namespace) -> None:
    payload = build_registry(args.root, project=args.project, encoders=parse_encoder_list(args.encoders))
    _print_payload(payload)


def _cmd_validate_layout(args: argparse.Namespace) -> None:
    payload = validate_layout(args.root, project=args.project, encoders=parse_encoder_list(args.encoders))
    _print_payload(payload)


def _cmd_promote_links(args: argparse.Namespace) -> None:
    payload = promote_links(args.root, project=args.project, encoders=parse_encoder_list(args.encoders))
    _print_payload(payload)


def _cmd_scan_h5_health(args: argparse.Namespace) -> None:
    payload = scan_h5_health(
        root=args.root,
        project=args.project,
        encoders=parse_encoder_list(args.encoders),
        source=args.source,
        legacy_root=args.legacy_root,
        out_dir=args.out_dir,
        stop_on_error=bool(args.stop_on_error),
    )
    _print_payload(payload)


def main() -> None:
    args = _build_parser().parse_args()
    args.handler(args)
