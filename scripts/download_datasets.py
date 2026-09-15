"""Dataset provenance guide and manual acquisition reference for remote sensing research.

IMPORTANT ACQUISITION POLICY:
The dataset is supplied manually by the researcher. The pipeline validates
the supplied dataset rather than downloading it automatically.

This script provides official citations, repository references, and placement instructions
for the user-provided datasets. It does NOT automatically download or overwrite files.

Usage:
    python3 scripts/download_datasets.py --info
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MANUAL_WORKFLOW_DOCUMENTATION = """
================================================================================
USER-PROVIDED DATASET INTEGRATION WORKFLOW
================================================================================

The RSICD dataset is supplied manually by the researcher.
The pipeline validates the supplied dataset rather than downloading it automatically.

Workflow steps:
1. User provides valid RSICD image files (and optional dataset_rsicd.json).
2. Place the dataset files into:
       data/raw/rsicd/images/      (or directly in data/raw/rsicd/)
       data/raw/rsicd/dataset_rsicd.json  (optional)
3. Run the objective integrity validator:
       python3 -m src.data.verify_rsicd
4. Inspect visual sample sanity check:
       python3 -m src.data.inspect_rsicd
5. If validation passes, generate structured metadata:
       python3 -c "from src.data.metadata_builder import MetadataBuilder; b = MetadataBuilder.build_for_rsicd_directory('data/raw/rsicd/images', 'data/raw/rsicd/dataset_rsicd.json'); b.save_json('data/metadata/rsicd/rsicd_metadata.json'); b.save_csv('data/metadata/rsicd/rsicd_metadata.csv'); print('Metadata generated successfully.')"
6. Preprocessing outputs will be generated into:
       data/processed/rsicd/
   Raw data inside data/raw/ remains strictly untouched.
================================================================================
"""

DATASET_PROVENANCE = {
    "RSICD": {
        "title": "Remote Sensing Image Captioning Dataset (RSICD)",
        "authors": "Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, and Xuelong Li",
        "publication": "IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2018",
        "doi": "10.1109/TGRS.2017.2776321",
        "official_repo": "https://github.com/201528014227051/RSICD_optimal",
        "community_repo": "https://huggingface.co/datasets/arampacha/rsicd",
        "reported_images": 10921,
        "reported_resolution": "224x224 RGB",
        "reported_categories": 30,
        "acquisition_mode": "Manual researcher provisioning into data/raw/rsicd/",
    },
    "LEVIR-CD": {
        "title": "Large-scale Building Change Detection Dataset (LEVIR-CD)",
        "authors": "Hao Chen and Zhenwei Shi",
        "publication": "Remote Sensing, MDPI (2020)",
        "doi": "10.3390/rs12101662",
        "official_site": "https://justchenhao.github.io/LEVIR/",
        "official_repo": "https://github.com/justchenhao/STANet",
        "reported_pairs": 637,
        "reported_resolution": "1024x1024 RGB (0.5m GSD)",
        "official_splits": "445 train / 64 val / 128 test",
        "acquisition_mode": "Manual researcher provisioning into data/raw/levir_cd/",
    }
}


def main():
    parser = argparse.ArgumentParser(description="Dataset provenance and manual integration guide")
    parser.add_argument("--info", action="store_true", default=True, help="Display provenance and workflow instructions")
    args = parser.parse_args()

    print(MANUAL_WORKFLOW_DOCUMENTATION)
    print("AUTHORITATIVE DATASET PROVENANCE:")
    print("-" * 80)
    for name, info in DATASET_PROVENANCE.items():
        print(f"\n[{name}]")
        for k, v in info.items():
            print(f"  {k}: {v}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
