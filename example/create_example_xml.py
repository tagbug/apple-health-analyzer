#!/usr/bin/env python3
"""
Create an example XML file with 10,000 diverse health records for testing.

This script extracts a representative sample of records from the real export.xml
file, ensuring coverage of different record types and categories.
"""

import sys
from collections import defaultdict
from pathlib import Path
from random import sample, seed
from typing import Any
from xml.etree import ElementTree as ET

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.xml_parser import StreamingXMLParser
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_example_xml(
  source_xml_path: Path,
  output_xml_path: Path,
  target_record_count: int = 10000,
  random_seed: int | None = None,
) -> dict[str, Any]:
  """
  Create an example XML file with diverse health records.

  Args:
      source_xml_path: Path to the source export.xml file
      output_xml_path: Path to save the example XML file
      target_record_count: Target number of records to include
      random_seed: Random seed for reproducible sampling. If None, uses current timestamp.

  Returns:
      Statistics about the created example file
  """
  if random_seed is None:
    # Use current timestamp for different results each time
    from time import time

    random_seed = int(time())

  seed(random_seed)  # For reproducible sampling

  logger.info(f"Creating example XML from {source_xml_path}")
  logger.info(f"Target record count: {target_record_count}")

  # Parse source XML and group records by type
  parser = StreamingXMLParser(source_xml_path)
  records_by_type = defaultdict(list)

  logger.info("Parsing source XML and grouping records by type...")

  record_count = 0
  for record in parser.parse_records():
    record_type = record.record_type
    records_by_type[record_type].append(record)
    record_count += 1

    if record_count % 50000 == 0:
      logger.info(f"Processed {record_count} records...")

  logger.info(
    f"Found {len(records_by_type)} record types with {record_count} total records"
  )

  # Calculate sampling strategy
  type_counts = {
    rtype: len(records) for rtype, records in records_by_type.items()
  }
  logger.info("Record type distribution:")
  for rtype, count in sorted(
    type_counts.items(), key=lambda x: x[1], reverse=True
  )[:10]:
    logger.info(f"  {rtype}: {count}")

  # Sample records to reach target count with diverse representation
  # Use a more efficient approach with indices to avoid expensive object comparisons
  selected_records = []
  selected_indices = {}  # Track selected record indices by type

  # Initialize tracking for each type
  for record_type in records_by_type:
    selected_indices[record_type] = set()

  remaining_slots = target_record_count

  # First, ensure we have at least some records from each type (up to a minimum)
  min_per_type = max(1, target_record_count // (len(records_by_type) * 2))

  for record_type, records in records_by_type.items():
    if len(records) <= min_per_type:
      # Take all records if we have fewer than minimum
      selected_records.extend(records)
      selected_indices[record_type].update(range(len(records)))
      remaining_slots -= len(records)
    else:
      # Sample minimum records from this type
      available_indices = list(range(len(records)))
      sampled_indices = sample(available_indices, min_per_type)
      sampled_records = [records[i] for i in sampled_indices]
      selected_records.extend(sampled_records)
      selected_indices[record_type].update(sampled_indices)
      remaining_slots -= min_per_type

  # Fill remaining slots proportionally by record type frequency
  if remaining_slots > 0:
    # Calculate remaining records per type based on original proportions
    total_remaining_records = sum(
      len(records) for records in records_by_type.values()
    ) - len(selected_records)
    if total_remaining_records > 0:
      for record_type, records in records_by_type.items():
        remaining_for_type = len(records) - len(selected_indices[record_type])
        if remaining_for_type > 0:
          # Calculate how many more to take from this type
          proportion = remaining_for_type / total_remaining_records
          take_count = min(
            remaining_for_type, int(remaining_slots * proportion)
          )

          if take_count > 0:
            available_indices = [
              i
              for i in range(len(records))
              if i not in selected_indices[record_type]
            ]
            if available_indices:
              sampled_indices = sample(
                available_indices, min(take_count, len(available_indices))
              )
              sampled_records = [records[i] for i in sampled_indices]
              selected_records.extend(sampled_records)
              selected_indices[record_type].update(sampled_indices)
              remaining_slots -= len(sampled_records)

              if remaining_slots <= 0:
                break

  # If we still don't have enough, take more from the most common types
  if len(selected_records) < target_record_count:
    logger.warning(
      f"Only collected {len(selected_records)} records, supplementing from common types"
    )
    for record_type in sorted(
      type_counts.keys(), key=lambda x: type_counts[x], reverse=True
    ):
      if len(selected_records) >= target_record_count:
        break

      records = records_by_type[record_type]
      available_indices = [
        i for i in range(len(records)) if i not in selected_indices[record_type]
      ]
      if available_indices:
        take_count = min(
          len(available_indices), target_record_count - len(selected_records)
        )
        sampled_indices = sample(available_indices, take_count)
        sampled_records = [records[i] for i in sampled_indices]
        selected_records.extend(sampled_records)
        selected_indices[record_type].update(sampled_indices)

  # Shuffle the final selection for better distribution
  from random import shuffle

  shuffle(selected_records)

  # Trim to exact target count if we have more
  if len(selected_records) > target_record_count:
    selected_records = selected_records[:target_record_count]

  logger.info(f"Selected {len(selected_records)} records for example XML")

  # Create XML structure
  logger.info("Creating XML structure...")

  # Create root element
  root = ET.Element("HealthData", {"locale": "zh_CN"})

  # Add export date
  from datetime import datetime

  export_date = ET.SubElement(
    root,
    "ExportDate",
    {"value": datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")},
  )

  # Add Me element (required by Apple Health format)
  me = ET.SubElement(root, "Me")
  ET.SubElement(
    me, "HKCharacteristicTypeIdentifierDateOfBirth"
  ).text = "1980-01-01"
  ET.SubElement(
    me, "HKCharacteristicTypeIdentifierBiologicalSex"
  ).text = "HKBiologicalSexMale"
  ET.SubElement(
    me, "HKCharacteristicTypeIdentifierBloodType"
  ).text = "HKBloodTypeNotSet"
  ET.SubElement(
    me, "HKCharacteristicTypeIdentifierFitzpatrickSkinType"
  ).text = "HKFitzpatrickSkinTypeNotSet"

  # Convert records back to XML elements
  record_elements = []
  for record in selected_records:
    elem = record_to_xml_element(record)
    if elem is not None:
      record_elements.append(elem)

  # Add records to root
  for elem in record_elements:
    root.append(elem)

  # Write XML file
  logger.info(f"Writing example XML to {output_xml_path}")
  tree = ET.ElementTree(root)
  tree.write(
    output_xml_path, encoding="utf-8", xml_declaration=True, method="xml"
  )

  # Collect statistics
  final_type_counts = defaultdict(int)
  for record in selected_records:
    final_type_counts[record.record_type] += 1

  stats = {
    "source_file": str(source_xml_path),
    "output_file": str(output_xml_path),
    "target_record_count": target_record_count,
    "actual_record_count": len(selected_records),
    "record_types_included": len(final_type_counts),
    "record_type_distribution": dict(final_type_counts),
    "random_seed": random_seed,
  }

  logger.info("Example XML creation completed!")
  logger.info(f"Final record count: {len(selected_records)}")
  logger.info(f"Record types included: {len(final_type_counts)}")

  return stats


def record_to_xml_element(record) -> ET.Element | None:
  """
  Convert a health record back to an XML element.

  Args:
      record: Health record instance

  Returns:
      XML element or None if conversion fails
  """
  try:
    # Create Record element
    elem = ET.Element("Record")

    # Set basic attributes
    if hasattr(record, "type"):
      elem.set("type", record.type)

    if hasattr(record, "source_name"):
      elem.set("sourceName", record.source_name)

    if hasattr(record, "source_version") and record.source_version:
      elem.set("sourceVersion", record.source_version)

    if hasattr(record, "device") and record.device:
      elem.set("device", record.device)

    if hasattr(record, "unit") and record.unit:
      elem.set("unit", record.unit)

    # Set value
    if hasattr(record, "value"):
      if isinstance(record.value, str):
        elem.set("value", record.value)
      else:
        elem.set("value", str(record.value))

    # Set dates
    date_format = "%Y-%m-%d %H:%M:%S %z"

    if hasattr(record, "creation_date") and record.creation_date:
      elem.set("creationDate", record.creation_date.strftime(date_format))

    if hasattr(record, "start_date") and record.start_date:
      elem.set("startDate", record.start_date.strftime(date_format))

    if hasattr(record, "end_date") and record.end_date:
      elem.set("endDate", record.end_date.strftime(date_format))

    # Add metadata
    if hasattr(record, "metadata") and record.metadata:
      for key, value in record.metadata.items():
        meta_elem = ET.SubElement(elem, "MetadataEntry")
        meta_elem.set("key", str(key))
        meta_elem.set("value", str(value))

    return elem

  except Exception as e:
    logger.warning(f"Failed to convert record to XML: {e}")
    return None


def main():
  """Main entry point."""
  import argparse

  parser = argparse.ArgumentParser(
    description="Create example XML file with diverse health records"
  )
  parser.add_argument(
    "--source",
    type=str,
    default="export_data/export.xml",
    help="Path to source export.xml file",
  )
  parser.add_argument(
    "--output",
    type=str,
    default="example/example.xml",
    help="Path to output example.xml file",
  )
  parser.add_argument(
    "--count",
    type=int,
    default=10000,
    help="Target number of records to include",
  )
  parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducible sampling"
  )

  args = parser.parse_args()

  source_path = Path(args.source)
  output_path = Path(args.output)

  if not source_path.exists():
    logger.error(f"Source file does not exist: {source_path}")
    sys.exit(1)

  # Create output directory if needed
  output_path.parent.mkdir(parents=True, exist_ok=True)

  try:
    stats = create_example_xml(source_path, output_path, args.count, args.seed)

    print("\n=== Example XML Creation Summary ===")
    print(f"Source file: {stats['source_file']}")
    print(f"Output file: {stats['output_file']}")
    print(f"Target records: {stats['target_record_count']}")
    print(f"Actual records: {stats['actual_record_count']}")
    print(f"Record types: {stats['record_types_included']}")
    print("\nTop record types in example:")
    for rtype, count in sorted(
      stats["record_type_distribution"].items(),
      key=lambda x: x[1],
      reverse=True,
    )[:10]:
      print(f"  {rtype}: {count}")

  except Exception as e:
    logger.error(f"Failed to create example XML: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
