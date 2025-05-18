## Installation

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

The basic usage is:

```
python webhook_catch_final_v3.py -l <log_file> -t <log_type> [options]
```

### Example

```
python webhook_catch_final_v3.py -l SAMPLE_DATA/RAW_APACHE_LOGS/access.log -t apache -o -p
```

### Options

- `-l, --log_file`: The raw log file (required)
- `-t, --log_type`: Log type: apache, http, nginx, or os_processes (required)
- `-e, --eps`: DBSCAN Epsilon value (max distance between two points)
- `-s, --min_samples`: Minimum number of points in a cluster
- `-j, --log_lines_limit`: Maximum number of log lines to consider
- `-y, --opt_lamda`: Optimization lambda step
- `-m, --minority_threshold`: Minority clusters threshold
- `-p, --show_plots`: Show informative plots
- `-o, --standardize_data`: Standardize feature values
- `-r, --report`: Create an HTML report
- `-z, --opt_silouhette`: Optimize DBSCAN silhouette coefficient
- `-b, --debug`: Activate debug logging
- `-c, --label_encoding`: Use label encoding instead of frequency encoding
- `-v, --find_cves`: Find CVEs related to attack traces
- `-n, --n_components`: Number of components for PCA (default: 2)

## Files

- `webhook_catch_final_v3.py`: The main script with all fixes applied
- `utilities_fixed.py`: Utility functions for parsing and processing logs
- `settings.conf`: Configuration file for log formats and feature definitions
- `test_final_fix.py`: Test script to validate the fixes

