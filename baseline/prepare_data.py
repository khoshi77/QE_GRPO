"""Snapshot existing evaluation parquet files; inference only uses local copies."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    sources = {"labels": "qe_wmt22_en_de", "xml_mt": "qe_wmt22_en_de_xml_mt"}
    entries = []
    for fmt, folder in sources.items():
        for split in ("dev", "test"):
            source = args.source_dir / folder / f"{split}.parquet"
            destination = args.output_dir / fmt / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Different data already exists: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            entries.append({"source": str(source.resolve()), "file": str(destination.relative_to(args.output_dir)),
                            "sha256": digest})
            print(destination)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
