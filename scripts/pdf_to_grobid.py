#!/usr/bin/env env python3

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from grobid_client.grobid_client import GrobidClient
except ImportError:
    print("Error: grobid-client-python is not installed.")
    print("Please install it with: pip install grobid-client-python")
    sys.exit(1)


def process_pdfs_with_grobid(
    input_folder: str,
    output_folder: str,
    config_path: str = "./config.json",
):
    """
    Process all PDF files in the input folder with GROBID and save TEI XML to output folder.

    Args:
        input_folder: Path to folder containing PDF files
        output_folder: Path to folder where XML files will be saved
        config_path: Path to GROBID client configuration file
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a directory.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    pdf_files = list(Path(input_folder).glob("*.pdf"))
    pdf_files.extend(list(Path(input_folder).glob("*.PDF")))

    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process")

    try:
        client = GrobidClient(config_path=config_path)
    except Exception as e:
        print(f"Error initializing GROBID client: {e}")
        print(f"Make sure the config file exists at '{config_path}'")
        sys.exit(1)

    # Create temporary directories for processing
    temp_input_dir = tempfile.mkdtemp()
    temp_output_dir = tempfile.mkdtemp()

    try:
        print("\nPreparing files for processing...")
        for pdf_file in pdf_files:
            pdf_filename = os.path.basename(pdf_file)
            temp_pdf_path = os.path.join(temp_input_dir, pdf_filename)
            shutil.copy2(pdf_file, temp_pdf_path)
            print(f"  - {pdf_filename}")

        # Process with GROBID
        print(f"\nProcessing {len(pdf_files)} PDF(s) with GROBID...")
        print(
            "This may take a while depending on the number and size of files...",
        )

        client.process(
            "processFulltextDocument",
            temp_input_dir,
            output=temp_output_dir,
        )

        print("\nSaving results...")
        processed_count = 0
        failed_count = 0

        for pdf_file in pdf_files:
            pdf_filename = os.path.basename(pdf_file)

            output_filename = (
                f"{os.path.splitext(pdf_filename)[0]}.grobid.tei.xml"
            )
            temp_output_path = os.path.join(temp_output_dir, output_filename)

            if os.path.exists(temp_output_path):
                final_output_path = os.path.join(
                    output_folder,
                    output_filename,
                )
                shutil.copy2(temp_output_path, final_output_path)
                print(f"  ✓ {pdf_filename} -> {output_filename}")
                processed_count += 1
            else:
                print(
                    f"  ✗ {pdf_filename} - Processing failed (no output generated)",
                )
                failed_count += 1

        # Summary
        print(f"\n{'=' * 60}")
        print("Processing complete!")
        print(f"Successfully processed: {processed_count}/{len(pdf_files)}")
        if failed_count > 0:
            print(f"Failed: {failed_count}/{len(pdf_files)}")
        print(f"Output saved to: {os.path.abspath(output_folder)}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)

    finally:
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process PDF files with GROBID and extract TEI XML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run pdf_to_grobid.py --input_path ./pdfs --output_path ./output
  uv run pdf_to_grobid.py --input_path ./pdfs --output_path ./output --config ./my_config.json

Note: Make sure GROBID server is running before executing this script.
See: https://grobid.readthedocs.io/en/latest/Grobid-service/
        """,
    )

    parser.add_argument(
        "--input_folder",
        help="Path to folder containing PDF files to process",
    )

    parser.add_argument(
        "--output_folder",
        help="Path to folder where XML output files will be saved",
    )

    parser.add_argument(
        "--config",
        default="./config.json",
        help="Path to GROBID client configuration file (default: ./config.json)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GROBID Batch PDF Processor")
    print("=" * 60)

    process_pdfs_with_grobid(
        args.input_folder,
        args.output_folder,
        args.config,
    )


if __name__ == "__main__":
    main()
