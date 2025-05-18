#!/usr/bin/env python
# Webhawk/Catch 2.0
# About: Use unsupervised learning to detect intrusion/suspicious activities in http logs
# Author: walid.daboubi@gmail.com
# Reviewed: 2023/08/06 - Flight Zurich/Las Vegas
# Fixed: 2025/05/18

import io
import kneed
import argparse
import pyfiglet
import termcolor
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.metrics
import matplotlib.pyplot as plt
import urllib3
import requests
import os
import sys
import logging
import time
import traceback

from utilities_fixed import *


def get_data(log_file, log_type, log_size_limit, features, encoding_type):
    """
    Converts a log file to a dataframe

    Parameters:
    log_file (str): The file which contains the logs
    log_type (str): The type of logs
    log_size_limit (int): Used to limit the number of lines to get from the log file
    features (str): A comma separated list of features
    encoding_type (str): label_encoding or fraction encoding

    Returns:
    pd.DataFrame or dict: A dataframe or dictionary containing the log data
    """
    if log_type == "os_processes":
        data = parse_process_file(log_file)
        data = pd.DataFrame.from_records(data)
        data = data.drop_duplicates(subset=["PID"], keep="last")
        data = data.set_index("PID")
        numerical_cols = list(data._get_numeric_data().columns)
        data = data[numerical_cols]
        data = data.drop(["PGRP", "PPID", "UID"], axis=1)
    else:
        try:
            encoded_logs = encode_log_file(log_file, log_type, encoding_type)
            data = encoded_logs
        except Exception as e:
            logging.error(f"Something went wrong encoding data: {str(e)}")
            logging.debug(traceback.format_exc())
            sys.exit(1)
        try:
            _, data_str = construct_enconded_data_file(encoded_logs, False)
        except Exception as e:
            logging.error(f"Something went wrong constructing data: {str(e)}")
            logging.debug(traceback.format_exc())
            sys.exit(1)
        # Get the raw log lines
        csvStringIO = io.StringIO(data_str)
        data = pd.read_csv(csvStringIO, sep=",").head(log_size_limit)
        data = data[features]
    return data


def plot_data(data, title):
    """
    Makes informative plots

    Parameters:
    data (list): The data to plot
    title (str): Title of the plot

    Returns:
    None
    """
    if len(data) == 2:
        # 2D informational plot
        logging.info(
            "{} Plotting an informative 2 dimensional visualisation".format(" " * 4)
        )
        fig = plt.figure()
        plt.plot(data[0], data[1], "ko")
        plt.title(title)
    if len(data) == 3:
        # 3D informational plot
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        logging.info(
            "{}Plotting an informative 3 dimensional visualisation".format(" " * 4)
        )
        ax.scatter3D(data[0], data[1], data[2], color="black")
        plt.title(title)
    plt.show()


def find_elements_by_cluster(labels):
    """
    Return the number of elements by cluster (including the outliers 'cluster')

    Parameters:
    labels (array): Cluster labels for each point

    Returns:
    dict: Dictionary with cluster labels as keys and number of elements as values
    """
    elements_by_cluster = {}
    for label in set(labels):
        elements_by_cluster[label] = np.count_nonzero(labels == label)
    elements_by_cluster = {
        k: v for k, v in sorted(elements_by_cluster.items(), key=lambda item: item[1])
    }
    return elements_by_cluster


def catch(labels, data, original_data, label, log_type):
    """
    Return a list of findings for a specific cluster

    Parameters:
    labels (array): Cluster labels for each point
    data (DataFrame): Dataframe with processed data
    original_data (DataFrame): Original dataframe with log lines
    label (int): The cluster label to find
    log_type (str): The type of logs

    Returns:
    list: List of findings
    """
    log_line_number = 0
    if label == -1:
        severity = "high"
    else:
        severity = "medium"
    findings = []

    for point_label in labels:
        # Adding the anomalous points to the findings
        if point_label == label:
            if not log_type == "os_processes":
                # Get the log line from the original data if possible
                log_line = None
                try:
                    if (
                        isinstance(original_data, pd.DataFrame)
                        and "log_line" in original_data.columns
                    ):
                        if log_line_number < len(original_data):
                            log_line = original_data.iloc[log_line_number]["log_line"]
                    elif isinstance(data, pd.DataFrame) and "log_line" in data.columns:
                        if log_line_number < len(data):
                            log_line = data.iloc[log_line_number]["log_line"]
                except Exception as e:
                    logging.debug(f"Error retrieving log line: {str(e)}")

                if log_line is None:
                    log_line = "Log line not available"

                finding = {
                    "log_line_number": log_line_number,
                    "log_line": log_line,
                    "severity": severity,
                }
            else:
                try:
                    pid = int(data.iloc[[log_line_number]].index.values[0])
                    finding_line = data.iloc[[log_line_number]].values[0]
                    column_names = data.columns.to_list()
                    process_details = get_process_details(pid)
                    finding = {
                        "pid": pid,
                        "log_line": list(zip(column_names, finding_line)),
                        "severity": severity,
                        "process_details": process_details,
                    }
                except Exception as e:
                    logging.error(f"Error processing finding for OS process: {str(e)}")
                    finding = {
                        "pid": "unknown",
                        "log_line": "Error processing finding",
                        "severity": severity,
                        "process_details": {},
                    }
            findings.append(finding)
        log_line_number += 1
    return findings


def print_findings(findings, log_type):
    """
    Print findings to the terminal

    Parameters:
    findings (list): List of findings
    log_type (str): The type of logs

    Returns:
    None
    """
    for finding in findings:
        if not log_type == "os_processes":
            logging.info(
                "\n\t/!\\ Webhawk {} - Possible anomalous behaviour detected at line:{}".format(
                    finding["severity"], finding["log_line_number"]
                )
            )
        else:
            logging.info(
                "\n\t/!\\ Webhawk {} - Possible anomalous behaviour detected at line:{}".format(
                    finding["severity"], finding["pid"]
                )
            )
        logging.info("\t{}".format(finding["log_line"]))


def plot_findings(dataframe, labels, save_at):
    """
    Plot findings

    Parameters:
    dataframe (DataFrame): Dataframe with processed data
    labels (array): Cluster labels for each point
    save_at (str): Path to save the plot, None to not save

    Returns:
    None
    """
    # Plot finddings
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(1, 0, len(unique_labels))]
    outliers_count = 0
    fig = plt.figure()

    for index, row in dataframe.iterrows():
        label = labels[index]
        # Plot outliers
        if label == -1:
            marker = "x"
            markersize = 10
            markeredgecolor = "r"
            outliers_count += 1
        # Plot other (minority cluster points)
        else:
            marker = "o"
            markersize = 6
            markeredgecolor = "black"
        plt.plot(
            row["pc_1"],
            row["pc_2"],
            marker,
            markerfacecolor=tuple(plt.cm.Spectral(np.linspace(label, 1, 1))[0]),
            markeredgecolor=markeredgecolor,
            markersize=markersize,
        )
    plt.title(
        "Webhawk/Catch - {} Possible attack traces detected".format(outliers_count)
    )
    if save_at != None:
        plt.savefig(save_at)
    plt.show()


def find_max_curvature_point(dataframe, plot):
    """
    Find the maximum curvature point among the sorted neighbors distance plot

    Parameters:
    dataframe (DataFrame): Dataframe with processed data
    plot (bool): Whether to plot the curvature

    Returns:
    float: Maximum curvature point
    """
    # Finding the nearest neighbors
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(dataframe)
    distances, indices = nbrs.kneighbors(dataframe)
    # Sorting the nearest neighbors distance
    sorted_idx = distances[:, 1].argsort()
    distances = distances[sorted_idx]
    indices = indices[sorted_idx]
    # Finding the maximum curvature point
    kl = kneed.KneeLocator(distances[:, 1], indices[:, 1], curve="convex", S=10.0)
    # Make plot if required
    if plot:
        plt.title("Sorted distance to nearest neighbors and max curvature")
        plt.plot(distances[:, 1])
        plt.axhline(kl.knee, 0, 1, label="max curve", color="black", linestyle="--")
        plt.show()
    return kl.knee


def optimize_silouhette_coefficient(max_curve, dataframe, lambda_value):
    """
    Optimize Epsilon to get the best BDSCAN silouhette Coefficient

    Parameters:
    max_curve (float): Maximum curvature point
    dataframe (DataFrame): Dataframe with processed data
    lambda_value (float): Lambda step value

    Returns:
    tuple: Best silhouette score and best epsilon value
    """
    current_eps = lambda_value
    best_silouhette = 0
    best_eps_for_silouhette = None
    while current_eps <= 1.5 * max_curve:
        dbscan = sklearn.cluster.DBSCAN(eps=current_eps)
        dbscan_model = dbscan.fit(dataframe)
        labels = dbscan_model.labels_
        if len(set(dbscan_model.labels_)) > 1:
            current_silouhette = sklearn.metrics.silhouette_score(dataframe, labels)
            if current_silouhette > best_silouhette:
                best_silouhette = current_silouhette
                best_eps_for_silouhette = current_eps
            logging.debug("\nCurvature:{}".format(current_eps))
            logging.debug("Silhouette:{}".format(current_silouhette))
        current_eps += lambda_value
    return best_silouhette, best_eps_for_silouhette


def get_minority_clusters(elements_by_cluster, threshold):
    """
    Return the clusters that include a number of point <= threshold

    Parameters:
    elements_by_cluster (dict): Dictionary with cluster labels and number of elements
    threshold (int): Maximum number of elements to consider a cluster as minority

    Returns:
    list: List of minority cluster labels
    """
    minority_clusters = []
    if -1 in elements_by_cluster:
        threshold += elements_by_cluster[-1]
        for key in elements_by_cluster:
            if key != -1 and elements_by_cluster[key] <= threshold:
                minority_clusters.append(key)
    return minority_clusters


def find_cves(findings):
    """
    Find CVEs related to findings

    Parameters:
    findings (list): List of findings

    Returns:
    list: List of findings enriched with CVEs
    """
    bad_chars = ["/", "?", "=", "&", "%", "#"]
    enriched_findings = []
    checked_candidate_strings = {}
    for finding in findings:
        # Only find CVE for the high severity findings
        if finding["severity"] == "high":
            finding["cve"] = ""
            try:
                final_requested_url = finding["log_line"].split('"')[1].split(" ")[1]
                # Replace 'bad' chars by spaces
                for bad_char in bad_chars:
                    final_requested_url = final_requested_url.replace(bad_char, " ")
                for candidate_string in final_requested_url.split(" "):
                    if len(candidate_string) >= 10:
                        try:
                            if candidate_string not in checked_candidate_strings:
                                checked_candidate_strings[candidate_string] = ""
                                response = requests.get(
                                    "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={}".format(
                                        candidate_string
                                    ),
                                    verify=False,
                                )

                                max_cve_by_candidate_string = 10
                                if (
                                    response.status_code == 200
                                    and "vulnerabilities" in response.json()
                                ):
                                    for vuln in response.json()["vulnerabilities"]:
                                        logging.info(
                                            "{} found".format(vuln["cve"]["id"])
                                        )
                                        max_cve_by_candidate_string -= 1
                                        if max_cve_by_candidate_string == 0:
                                            break
                                        finding["cve"] += vuln["cve"]["id"] + " "
                                        checked_candidate_strings[candidate_string] += (
                                            vuln["cve"]["id"] + " "
                                        )
                                    # Sleep to ne be blocked by services.nvd.nist.gov
                                    time.sleep(5)
                            else:
                                finding["cve"] += (
                                    checked_candidate_strings[candidate_string] + " "
                                )
                        except Exception as e:
                            logging.warning(f"Error searching for CVE: {str(e)}")
            except Exception as e:
                logging.warning(f"Error parsing log line for CVE search: {str(e)}")
        enriched_findings.append(finding)
    return enriched_findings


def preprocess_dataframe(dataframe, standardize=False):
    """
    Preprocess the dataframe for machine learning

    Parameters:
    dataframe (DataFrame): Original dataframe
    standardize (bool): Whether to standardize the data

    Returns:
    tuple: Processed dataframe, log lines, and original indices
    """
    # Save original indices to maintain relationship after transformations
    original_indices = None

    # Save the log_line column for later use
    log_lines = None
    if isinstance(dataframe, pd.DataFrame) and "log_line" in dataframe.columns:
        log_lines = dataframe["log_line"].copy()
        original_indices = dataframe.index
        logging.info("Saved log_line column for later reference")
        # Remove the log_line column as it's a string and can't be used for PCA
        dataframe = dataframe.drop("log_line", axis=1)

    # Check for other non-numeric columns and convert or remove them
    if isinstance(dataframe, pd.DataFrame):
        for col in dataframe.columns:
            if dataframe[col].dtype == "object":
                # If it's a categorical column, we could use one-hot encoding
                # For simplicity, we'll just drop it
                logging.info(f"Dropping non-numeric column: {col}")
                dataframe = dataframe.drop(col, axis=1)

        # Make sure all data is numeric
        dataframe = dataframe.apply(pd.to_numeric, errors="coerce")

        # Drop any rows with NaN values that might have been created
        if dataframe.isnull().any().any():
            logging.info("Dropping rows with NaN values")
            # Save which indices are being kept
            kept_indices = dataframe.dropna().index
            dataframe = dataframe.dropna()

            # Update log_lines to match the filtered dataframe
            if log_lines is not None:
                log_lines = log_lines.loc[kept_indices]

    # Standardize data if requested
    if standardize:
        logging.info("Standardizing data...")
        scaler = sklearn.preprocessing.StandardScaler()
        if isinstance(dataframe, pd.DataFrame):
            # Save the index
            df_index = dataframe.index
            # Fit transform
            scaled_data = scaler.fit_transform(dataframe)
            # Create a new dataframe with the original index
            dataframe = pd.DataFrame(
                scaled_data, index=df_index, columns=dataframe.columns
            )
        else:
            dataframe = scaler.fit_transform(dataframe)
            dataframe = pd.DataFrame(dataframe)

    return dataframe, log_lines, original_indices


def apply_pca(dataframe, log_lines, n_components=2):
    """
    Apply PCA to the dataframe

    Parameters:
    dataframe (DataFrame): Preprocessed dataframe
    log_lines (Series): Log lines series
    n_components (int): Number of components for PCA

    Returns:
    DataFrame: Dataframe with PCA applied
    """
    logging.info(
        f"Applying PCA for dimensionality reduction with {n_components} components..."
    )
    pca = sklearn.decomposition.PCA(n_components=n_components)

    # Save the index before PCA
    if isinstance(dataframe, pd.DataFrame):
        df_index = dataframe.index
        principal_components_df = pca.fit_transform(dataframe)
        # Create a new dataframe with the original index
        column_names = [f"pc_{i+1}" for i in range(n_components)]
        dataframe = pd.DataFrame(
            data=principal_components_df, columns=column_names, index=df_index
        )
    else:
        principal_components_df = pca.fit_transform(dataframe)
        column_names = [f"pc_{i+1}" for i in range(n_components)]
        dataframe = pd.DataFrame(data=principal_components_df, columns=column_names)

    # Add back the log_line column for reference in findings
    if log_lines is not None:
        try:
            # Make sure the indices align
            common_indices = dataframe.index.intersection(log_lines.index)
            dataframe = dataframe.loc[common_indices]
            log_lines = log_lines.loc[common_indices]
            dataframe["log_line"] = log_lines
            logging.info("Restored log_line column to results dataframe")
        except Exception as e:
            logging.warning(f"Error adding log_line column back: {str(e)}")

    logging.info(f"DataFrame shape after PCA: {dataframe.shape}")
    return dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_file", help="The raw log file", required=True)
    parser.add_argument(
        "-t", "--log_type", help="apache, http, nginx or os_processes", required=True
    )
    parser.add_argument(
        "-e",
        "--eps",
        help="DBSCAN Epsilon value (Max distance between two points)",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--min_samples",
        help="Minimum number of points with the same cluster. The default value is 2",
        required=False,
    )
    parser.add_argument(
        "-j",
        "--log_lines_limit",
        help="The maximum number of log lines of consider",
        required=False,
    )
    parser.add_argument(
        "-y", "--opt_lamda", help="Optimization lambda step", required=False
    )
    parser.add_argument(
        "-m", "--minority_threshold", help="Minority clusters threshold", required=False
    )
    parser.add_argument(
        "-p", "--show_plots", help="Show informative plots", action="store_true"
    )
    parser.add_argument(
        "-o",
        "--standardize_data",
        help="Standardize feature values",
        action="store_true",
    )
    parser.add_argument(
        "-r", "--report", help="Create a HTML report", action="store_true"
    )
    parser.add_argument(
        "-z", "--opt_silouhette", help="Optimize DBSCAN silouhette", action="store_true"
    )
    parser.add_argument(
        "-b", "--debug", help="Activate debug logging", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--label_encoding",
        help="Use label encoding instead of frequeny encoding to encode categorical features",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--find_cves",
        help="Find the CVE(s) that are related to the attack traces",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--n_components",
        help="Number of components for PCA. Default is 2",
        type=int,
        default=2,
    )
    urllib3.disable_warnings()

    # Get parameters
    args = vars(parser.parse_args())

    # Set up logging
    logging_level = logging.DEBUG if args["debug"] else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    LOG_LINES_LIMIT = (
        int(args["log_lines_limit"]) if args["log_lines_limit"] is not None else 1000000
    )
    LAMBDA = float(args["opt_lamda"]) if args["opt_lamda"] is not None else 0.01
    THRESHOLD = int(args["minority_threshold"]) if args["minority_threshold"] else 5
    N_COMPONENTS = args["n_components"]

    encoding_type = (
        "label_encoding" if args["label_encoding"] == True else "fraction_encoding"
    )

    FEATURES = [
        "params_number",
        "length",
        "upper_cases",
        "lower_cases",
        "special_chars",
        "url_depth",
        "user_agent",
        "http_query",
        "ip",
        "return_code",
        "size",
        "log_line",
    ]

    print("\n")
    print(
        (
            termcolor.colored(
                pyfiglet.figlet_format(
                    "Webhawk / Catch 2.0", font="banner3", width=600
                ),
                color="yellow",
            )
        )
    )

    logging.info("\n> Webhawk Catch 2.0")
    logging.info("{}The input log file is {}".format(" " * 4, args["log_file"]))
    logging.info("{}Log format is set to {}".format(" " * 4, args["log_type"]))
    logging.info("{}Demo plotting is set to {}".format(" " * 4, args["show_plots"]))
    logging.info(
        "{}Features standarization is set to {}".format(
            " " * 4, args["standardize_data"]
        )
    )
    logging.info("{}PCA components: {}".format(" " * 4, N_COMPONENTS))

    # Get data
    logging.info("\n> Data reading started")

    try:
        data = get_data(
            args["log_file"], args["log_type"], LOG_LINES_LIMIT, FEATURES, encoding_type
        )
    except Exception as e:
        logging.error(f"Error reading data: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1)

    # Keep a copy of raw data for reference
    original_data = data

    logging.info("Data loaded. Processing...")

    # Convert to a dataframe
    if args["log_type"] != "os_processes":
        # Check if data is already a dataframe
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            try:
                dataframe = pd.DataFrame.from_dict(data, orient="index")
            except Exception as e:
                logging.error(f"Error converting data to DataFrame: {str(e)}")
                logging.debug(traceback.format_exc())
                sys.exit(1)
    else:
        dataframe = data

    logging.info(f"DataFrame shape: {dataframe.shape}")

    try:
        # Preprocess the dataframe
        dataframe, log_lines, original_indices = preprocess_dataframe(
            dataframe, standardize=args["standardize_data"]
        )

        # Show informative data plots
        if args["show_plots"]:
            try:
                if isinstance(dataframe, pd.DataFrame) and dataframe.shape[1] >= 2:
                    plot_data(
                        [dataframe.iloc[:, 0], dataframe.iloc[:, 1]], "Original data"
                    )
            except Exception as e:
                logging.warning(f"Could not create plot for original data: {str(e)}")

        # Apply PCA
        dataframe = apply_pca(dataframe, log_lines, n_components=N_COMPONENTS)

        if args["show_plots"]:
            try:
                if "pc_1" in dataframe.columns and "pc_2" in dataframe.columns:
                    plot_data(
                        [dataframe["pc_1"], dataframe["pc_2"]],
                        "Data after dimensiality reduction using PCA",
                    )
            except Exception as e:
                logging.warning(f"Error plotting PCA data: {str(e)}")

        # Getting or setting epsilon
        if args["eps"] is None:
            try:
                pca_columns = [f"pc_{i+1}" for i in range(N_COMPONENTS)]
                pca_columns = [col for col in pca_columns if col in dataframe.columns]
                if len(pca_columns) >= 2:
                    max_curve_point = find_max_curvature_point(
                        dataframe[pca_columns], args["show_plots"]
                    )
                    logging.info(
                        "\n{}The maximum curve point is {}".format(
                            4 * " ", max_curve_point
                        )
                    )
                    selected_eps = max_curve_point
                else:
                    logging.warning(
                        "Not enough components for finding max curvature point"
                    )
                    selected_eps = 0.5  # Default value
            except Exception as e:
                logging.warning(f"Error finding max curvature point: {str(e)}")
                selected_eps = 0.5  # Default value
        else:
            selected_eps = float(args["eps"])

        if args["opt_silouhette"]:
            try:
                pca_columns = [f"pc_{i+1}" for i in range(N_COMPONENTS)]
                pca_columns = [col for col in pca_columns if col in dataframe.columns]
                if len(pca_columns) >= 2:
                    best_silouhette, best_eps_for_silouhette = (
                        optimize_silouhette_coefficient(
                            selected_eps, dataframe[pca_columns], LAMBDA
                        )
                    )
                    logging.info(
                        "\n{}Best eps for DBSCAN Silouhette: {}".format(
                            4 * " ", best_eps_for_silouhette
                        )
                    )
                    logging.info(
                        "{}Best silouhette coefficient: {}".format(
                            4 * " ", best_silouhette
                        )
                    )
                    selected_eps = best_eps_for_silouhette
                else:
                    logging.warning(
                        "Not enough components for optimizing silhouette coefficient"
                    )
            except Exception as e:
                logging.warning(f"Error optimizing silhouette coefficient: {str(e)}")

        logging.info(
            "{}The value {} will be used as final DBSCAN Epsilon".format(
                4 * " ", selected_eps
            )
        )

        logging.info("\n> Starting detection..")
        # Use dbscan for clustering and train the model

        if args["eps"] is None and args["min_samples"] is None:
            dbscan = sklearn.cluster.DBSCAN(eps=selected_eps, min_samples=2)
        elif args["eps"] is None:
            dbscan = sklearn.cluster.DBSCAN(
                eps=selected_eps, min_samples=int(args["min_samples"])
            )
        elif args["min_samples"] is None:
            dbscan = sklearn.cluster.DBSCAN(eps=selected_eps, min_samples=2)
        else:
            dbscan = sklearn.cluster.DBSCAN(
                eps=selected_eps, min_samples=int(args["min_samples"])
            )

        try:
            # Use only PCA components for clustering
            pca_columns = [f"pc_{i+1}" for i in range(N_COMPONENTS)]
            pca_columns = [col for col in pca_columns if col in dataframe.columns]
            dbscan_model = dbscan.fit(dataframe[pca_columns])
        except Exception as e:
            logging.error(f"Error in DBSCAN fitting: {str(e)}")
            logging.debug(traceback.format_exc())
            sys.exit(1)

        # Check the number of labels (if 1 then try without EPS value)
        if len(set(dbscan_model.labels_)) == 1:
            logging.info(
                "\nOnly ONE cluster found. Maybe an anomaly free training dataset?"
            )
            logging.info("Re-running DBSCAN without setting EPS")
            dbscan = sklearn.cluster.DBSCAN()
            dbscan_model = dbscan.fit(dataframe[pca_columns])
        if len(set(dbscan_model.labels_)) == 1:
            logging.info("\nStill only ONE cluster found. Probably anomaly free logs.")

        # Get point labels
        labels = dbscan_model.labels_

        elements_by_cluster = find_elements_by_cluster(labels)
        number_of_clusters = len(elements_by_cluster.values())

        # Get top minority clusters
        minority_clusters = get_minority_clusters(elements_by_cluster, THRESHOLD)

        # Outliers are considred as high severity findings
        # Keep a copy of the original data with log_lines
        if isinstance(original_data, dict):
            try:
                original_data = pd.DataFrame.from_dict(original_data, orient="index")
            except Exception as e:
                logging.warning(
                    f"Could not convert original data to DataFrame: {str(e)}"
                )

        high_findings = catch(labels, dataframe, original_data, -1, args["log_type"])
        if len(high_findings) > 0:
            logging.info(
                "\n> {} high severity finding(s) detected".format(len(high_findings))
            )
            print_findings(high_findings, args["log_type"])

        # Points belonging to minority clusters are considred as medium severity findings
        medium_findings = []
        for label in minority_clusters:
            medium_findings += catch(
                labels, dataframe, original_data, label, args["log_type"]
            )

        if len(medium_findings) > 0:
            logging.info(
                "\n> {} medium severity finding(s) detected".format(
                    len(medium_findings)
                )
            )
            print_findings(medium_findings, args["log_type"])

        all_findings = high_findings + medium_findings

        if args["find_cves"]:
            all_findings = find_cves(all_findings)

        # Number of clusters in labels, ignoring noise if present.
        n_noise = list(labels).count(-1)

        logging.info("\nEstimated number of clusters: %d" % len(set(labels)))
        logging.info("Estimated number of outliers/anomalous points: %d" % n_noise)

        if (
            len(set(labels)) > 1
        ):  # Only calculate silhouette score if more than one cluster
            try:
                silhouette_score = sklearn.metrics.silhouette_score(
                    dataframe[pca_columns], labels
                )
                logging.info("DBSCAN Silhouette Coefficient: %0.3f" % silhouette_score)
            except Exception as e:
                logging.warning(f"Could not calculate silhouette score: {str(e)}")

        if args["report"] and len(all_findings) > 0:
            scan_dir = "./SCANS"
            if not os.path.isdir(scan_dir):
                os.makedirs(scan_dir)
            gen_report(all_findings, args["log_file"], args["log_type"])
            # Save finding plots to the filesystem
            if len(all_findings) > 0:
                try:
                    plot_findings(
                        dataframe,
                        labels,
                        "./SCANS/scan_plot_{}.png".format(
                            args["log_file"].split("/")[-1].replace(".", "_")
                        ),
                    )
                except Exception as e:
                    logging.warning(f"Error saving plot: {str(e)}")

        if args["show_plots"]:
            # Plot finddings
            try:
                plot_findings(dataframe, labels, None)
            except Exception as e:
                logging.warning(f"Error displaying plot: {str(e)}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
