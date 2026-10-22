"""Validate a local ONNX model and save an opset-compatible copy, preserving source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile


def external_locations(message):
    if hasattr(message, "external_data"):
        for item in message.external_data:
            if item.key == "location":
                yield item.value
    if hasattr(message, "ListFields"):
        for field, value in message.ListFields():
            if field.message_type is not None:
                repeated = field.is_repeated if hasattr(field, "is_repeated") else field.label == 3
                children = value if repeated else [value]
                for child in children:
                    yield from external_locations(child)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    try:
        if not source.is_file() or source.suffix.lower() != ".onnx":
            raise ValueError("input must be an existing .onnx file")
        if args.opset < 1 or output == source or output.exists():
            raise ValueError("positive opset and a new output path distinct from input required")
        import onnx
        model = onnx.load(str(source), load_external_data=False)
        for location in external_locations(model):
            target = (source.parent / location).resolve()
            if (Path(location).is_absolute() or PureWindowsPath(location).is_absolute() or
                    not target.is_relative_to(source.parent) or not target.is_file()):
                raise ValueError(f"external tensor data must exist within input directory: {location}")
        onnx.external_data_helper.load_external_data_for_model(model, str(source.parent))
        onnx.checker.check_model(str(source))
        versions = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
        if len(versions) != 1:
            raise ValueError("model must have one standard ONNX opset import")
        if versions[0] != args.opset:
            model = onnx.version_converter.convert_version(model, args.opset)
        output.parent.mkdir(parents=True, exist_ok=True)
        data_name = output.name + ".data"
        if (output.parent / data_name).exists():
            raise ValueError(f"output external data already exists: {data_name}")
        with tempfile.TemporaryDirectory(prefix="onnx-prepare-", dir=output.parent) as tmp:
            staged = Path(tmp) / output.name
            onnx.save_model(model, str(staged), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
            onnx.checker.check_model(str(staged))
            external = Path(tmp) / data_name
            if external.exists():
                shutil.move(str(external), str(output.parent / data_name))
            shutil.move(str(staged), str(output))
        print(json.dumps({"input": str(source), "output": str(output), "opset": args.opset, "onnx_checker": "ok"}))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"ONNX preparation failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
