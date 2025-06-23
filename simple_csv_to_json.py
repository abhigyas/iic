#!/usr/bin/env python3
"""
Simple CSV to JSON Converter

A lightweight script to quickly convert CSV files to JSON format.
"""

import csv
import json
import sys


def convert_csv_to_json(csv_file_path, json_file_path=None):
    """
    Convert a CSV file to JSON format.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path to the output JSON file (optional)
    
    Returns:
        list: The converted data as a list of dictionaries
    """
    data = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                # Clean up any extra whitespace
                cleaned_row = {key.strip(): value.strip() for key, value in row.items()}
                data.append(cleaned_row)
        
        print(f"Successfully read {len(data)} rows from {csv_file_path}")
        
        # Save to JSON file if path is provided
        if json_file_path:
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            print(f"JSON data saved to {json_file_path}")
        
        return data
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_csv_to_json.py <csv_file> [output_json_file]")
        print("Example: python simple_csv_to_json.py cyber_data.csv cyber_data.json")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else csv_file.replace('.csv', '.json')
    
    result = convert_csv_to_json(csv_file, json_file)
    
    if result:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
