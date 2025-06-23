#!/usr/bin/env python3
"""
CSV to JSON Converter Script

This script converts CSV files to JSON format with various output options.
It can handle large files efficiently and provides multiple JSON output formats.
"""

import csv
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


def csv_to_json_array(csv_file_path: str, output_file_path: str = None, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Convert CSV to JSON array format.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output JSON file (optional)
        encoding: File encoding (default: utf-8)
    
    Returns:
        List of dictionaries representing the CSV data
    """
    data = []
    
    try:
        with open(csv_file_path, 'r', encoding=encoding, newline='') as csv_file:
            # Use csv.Sniffer to detect delimiter
            sample = csv_file.read(1024)
            csv_file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            
            for row_num, row in enumerate(csv_reader, start=1):
                # Clean up any extra whitespace in values
                cleaned_row = {key.strip(): value.strip() if isinstance(value, str) else value 
                             for key, value in row.items() if key is not None}
                data.append(cleaned_row)
                
                # Progress indicator for large files
                if row_num % 1000 == 0:
                    print(f"Processed {row_num} rows...")
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file with {encoding} encoding. Try a different encoding.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Save to file if output path is provided
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            print(f"JSON file saved to: {output_file_path}")
        except Exception as e:
            print(f"Error writing JSON file: {e}")
            sys.exit(1)
    
    return data


def csv_to_json_lines(csv_file_path: str, output_file_path: str, encoding: str = 'utf-8') -> None:
    """
    Convert CSV to JSON Lines format (one JSON object per line).
    This is useful for large datasets and streaming processing.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output JSONL file
        encoding: File encoding (default: utf-8)
    """
    try:
        with open(csv_file_path, 'r', encoding=encoding, newline='') as csv_file, \
             open(output_file_path, 'w', encoding='utf-8') as json_file:
            
            # Use csv.Sniffer to detect delimiter
            sample = csv_file.read(1024)
            csv_file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            
            for row_num, row in enumerate(csv_reader, start=1):
                # Clean up any extra whitespace in values
                cleaned_row = {key.strip(): value.strip() if isinstance(value, str) else value 
                             for key, value in row.items() if key is not None}
                
                json_file.write(json.dumps(cleaned_row, ensure_ascii=False) + '\n')
                
                # Progress indicator for large files
                if row_num % 1000 == 0:
                    print(f"Processed {row_num} rows...")
        
        print(f"JSON Lines file saved to: {output_file_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def csv_to_nested_json(csv_file_path: str, output_file_path: str, group_by_column: str, encoding: str = 'utf-8') -> None:
    """
    Convert CSV to nested JSON grouped by a specific column.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output JSON file
        group_by_column: Column name to group the data by
        encoding: File encoding (default: utf-8)
    """
    grouped_data = {}
    
    try:
        with open(csv_file_path, 'r', encoding=encoding, newline='') as csv_file:
            # Use csv.Sniffer to detect delimiter
            sample = csv_file.read(1024)
            csv_file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            
            # Check if the group_by_column exists
            if group_by_column not in csv_reader.fieldnames:
                print(f"Error: Column '{group_by_column}' not found in CSV file.")
                print(f"Available columns: {', '.join(csv_reader.fieldnames)}")
                sys.exit(1)
            
            for row_num, row in enumerate(csv_reader, start=1):
                # Clean up any extra whitespace in values
                cleaned_row = {key.strip(): value.strip() if isinstance(value, str) else value 
                             for key, value in row.items() if key is not None}
                
                group_key = cleaned_row.get(group_by_column, 'Unknown')
                
                if group_key not in grouped_data:
                    grouped_data[group_key] = []
                
                grouped_data[group_key].append(cleaned_row)
                
                # Progress indicator for large files
                if row_num % 1000 == 0:
                    print(f"Processed {row_num} rows...")
        
        # Save grouped data to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(grouped_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Nested JSON file saved to: {output_file_path}")
        print(f"Data grouped by '{group_by_column}' with {len(grouped_data)} groups")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to JSON format with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion to JSON array
  python csv_to_json_converter.py cyber_data.csv

  # Convert to JSON array with custom output file
  python csv_to_json_converter.py cyber_data.csv -o cyber_data.json

  # Convert to JSON Lines format (one JSON object per line)
  python csv_to_json_converter.py cyber_data.csv -f jsonl -o cyber_data.jsonl

  # Convert to nested JSON grouped by country
  python csv_to_json_converter.py cyber_data.csv -f nested -g Country -o cyber_data_by_country.json

  # Specify encoding for the CSV file
  python csv_to_json_converter.py cyber_data.csv -e iso-8859-1
        """
    )
    
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output JSON file')
    parser.add_argument('-f', '--format', choices=['json', 'jsonl', 'nested'], default='json',
                        help='Output format: json (array), jsonl (lines), nested (grouped)')
    parser.add_argument('-g', '--group-by', help='Column name to group by (required for nested format)')
    parser.add_argument('-e', '--encoding', default='utf-8', help='File encoding (default: utf-8)')
    parser.add_argument('--preview', action='store_true', help='Preview first 5 rows without saving')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Generate output filename if not provided
    if not args.output and not args.preview:
        input_path = Path(args.input_file)
        if args.format == 'jsonl':
            args.output = str(input_path.with_suffix('.jsonl'))
        else:
            args.output = str(input_path.with_suffix('.json'))
    
    # Validate nested format requirements
    if args.format == 'nested' and not args.group_by:
        print("Error: --group-by is required when using nested format.")
        sys.exit(1)
    
    print(f"Converting '{args.input_file}' to {args.format.upper()} format...")
    
    try:
        if args.preview:
            # Preview mode - show first 5 rows
            data = csv_to_json_array(args.input_file, encoding=args.encoding)
            print("\nPreview of first 5 rows:")
            print(json.dumps(data[:5], indent=2, ensure_ascii=False))
            print(f"\nTotal rows: {len(data)}")
            
        elif args.format == 'json':
            csv_to_json_array(args.input_file, args.output, args.encoding)
            
        elif args.format == 'jsonl':
            csv_to_json_lines(args.input_file, args.output, args.encoding)
            
        elif args.format == 'nested':
            csv_to_nested_json(args.input_file, args.output, args.group_by, args.encoding)
        
        print("Conversion completed successfully!")
        
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
        sys.exit(1)


if __name__ == '__main__':
    main()
