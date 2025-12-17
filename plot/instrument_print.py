import os
import sys
import re
import argparse
import tarfile
import numpy as np
import pandas as pd

def struct_def(struct: dict, sections: list):
    if len(sections) <= 0:
        return

# Funzione per formattare i tempi in unità leggibili
def format_time_units_ns(value):
    """Format nanosecond values into ns / µs / ms without unnecessary .0."""
    if value < 1_000:  # < 1 µs
        return f"{int(value)}ns" if value.is_integer() else f"{value:.1f} ns"
    elif value < 1_000_000:  # < 1 ms
        val = value / 1_000
        return f"{int(val)}µs" if val.is_integer() else f"{val:.1f} µs"
    else:
        val = value / 1_000_000
        return f"{int(val)}ms" if val.is_integer() else f"{val:.1f} ms"
    
def human_readable_size(num_bytes):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num_bytes < 1024:
            return f"{int(num_bytes)} {unit}"
        num_bytes /= 1024
    return f"{int(num_bytes)} PiB"    

def print_table(df_filtered: pd.DataFrame):
    for algo_name, grouped_df in df_filtered.sort_values('algo_name').groupby('algo_name'):
        print("-" * 60)
        print(algo_name)
        print("-" * 60)
        grouped_df = grouped_df.sort_values(by='buffer_size').dropna(axis=1, how='all')
        
        colonne_mean = grouped_df.filter(regex='/mean$').columns

        colonne_da_calcolare = colonne_mean.drop('rank0/mean')

        df_percentuali = grouped_df[colonne_da_calcolare].div(grouped_df['rank0/mean'], axis=0) * 100

        df_percentuali.columns = [f"{"/".join(c.split("/")[:-1])}" for c in df_percentuali.columns]

        risultato = pd.concat([
            grouped_df['buffer_size'],
            grouped_df.rename(columns={'rank0/mean': 'time'})['time'], 
            df_percentuali
        ], axis=1)
        
        print(risultato.to_string(index=False,float_format='{:.2f}%'.format, formatters={'time':format_time_units_ns,'buffer_size':human_readable_size}))
        print()

def print_struct(df_filtered: pd.DataFrame):
    for algo_name, grouped_df in df_filtered.sort_values('algo_name').groupby('algo_name'):
        print("=" * 60)
        print(algo_name)
        print("=" * 60)
        grouped_df = grouped_df.sort_values(by='buffer_size').dropna(axis=1, how='all')
        
        colonne_mean = grouped_df.filter(regex='/mean$').columns

        colonne_da_calcolare = colonne_mean.drop('rank0/mean')

        df_percentuali = grouped_df[colonne_da_calcolare].div(grouped_df['rank0/mean'], axis=0) * 100

        df_percentuali.columns = [f"{"/".join(c.split("/")[:-1])}" for c in df_percentuali.columns]

        risultato = pd.concat([
            grouped_df['buffer_size'],
            grouped_df.rename(columns={'rank0/mean': 'time'})['time'], 
            df_percentuali
        ], axis=1)
        
        for index, row in risultato.iterrows():
            print(f"buffer size: {human_readable_size(row['buffer_size'])}")
            print(f"total time: {format_time_units_ns(row['time'])}")
            print()
            print(f"   {"section":<37}| time %")
            print("-" * 50)
            for colonna, valore in row.items():
                if colonna in ['buffer_size', 'time']: continue
                data_print = f"   {"  " * (len(colonna.split("/")) - 1)}{colonna}"
                print(f"{data_print:<40}| {valore:.2f}%")
            print("-" * 50, end="\n\n")
        print()
        
def print_struct_condensed(df_filtered: pd.DataFrame):
    for algo_name, grouped_df in df_filtered.sort_values('algo_name').groupby('algo_name'):
        print("=" * 85)
        print(algo_name)
        print("=" * 85)
        grouped_df = grouped_df.sort_values(by='buffer_size').dropna(axis=1, how='all')
        
        colonne_mean = grouped_df.filter(regex='/mean$').columns

        colonne_da_calcolare = colonne_mean.drop('rank0/mean')

        df_percentuali = grouped_df[colonne_da_calcolare].div(grouped_df['rank0/mean'], axis=0) * 100

        df_percentuali.columns = [f"{"/".join(c.split("/")[:-1])}" for c in df_percentuali.columns]

        risultato = pd.concat([
            grouped_df['buffer_size'],
            grouped_df.rename(columns={'rank0/mean': 'time'})['time'], 
            df_percentuali
        ], axis=1)
        
        print(f"buffer size: {"".join([f"{human_readable_size(val):8} " for val in risultato['buffer_size']])}")
        print(f"total time:  {"".join([f"{format_time_units_ns(val):8} " for val in risultato['time']])}", end="\n\n")
        print(f"   {"section":<37}| time % for every size")
        print(f"   {" " * 37}| {"|".join([f"{human_readable_size(val):8}" for val in risultato['buffer_size']])}")
        print("-" * 85)
        for colonna, valore in risultato.items():
            if colonna in ['buffer_size', 'time']: continue
            data_print = f"   {"  " * (len(colonna.split("/")) - 1)}{colonna}"
            print(f"{data_print:<40}| {"|".join([f"{val:6.2f}% " for val in valore])}")
        print("-" * 85, end="\n\n")

def main():
    parser = argparse.ArgumentParser(description="Instrument data print")
    parser.add_argument("--summary-file", required=True, help="Path to aggregated summary CSV.")
    args = parser.parse_args()

    # Ensure the path follows the expected format: results/<system>/<timestamp>
    this_agregated_file = os.path.normpath(args.summary_file)
    parts = this_agregated_file.split(os.sep)
    if len(parts) < 4 or parts[-4] != "results":
        print(f"Invalid result directory structure: {this_agregated_file}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(this_agregated_file, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        print(f"Empty data error for file: {this_agregated_file}", file=sys.stderr)
        return None
    
    df_filtered = df[df['gpu_awareness'] == 'yes']
    print_struct_condensed(df_filtered)
            
            
if __name__ == '__main__':
    main()