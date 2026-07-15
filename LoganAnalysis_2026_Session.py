'''
*****************************************************************#
Title                            Logan Analysis
Purpose                          Compiles dataset from
                                 multiple log files
                                 and last backup of each month
Creation Date                    08/27/2020
Created by                       John
Modified by                      Derek Vincent Taylor, Steve Flores, Jason R. Bock
Last Modified                    04/20/2026 by Jason R. Bock
Changelog                        
1. Mapping of image types has been coded in a unified fashion, by semantic, spatial, and temporal distance. Previously, "hard" and "easy" did not describe targets or foils, respectively, and "high" and "low" did not describe temporal distance.
2. Variables are recorded for raw counts and percent correct and incorrect, unrounded for use in derived values
3. Code has been cleaned and streamlined throughout.
4. All domains are written into a single dataframe and subsequently a single Excel sheet, with columns for each domain and creates a new row if the participant does not yet exist or merges into existing row
5. All three domains have Accuracy Overall and Accuracy Similar, a new metric that accounts for correct responses over total responses for only target/simhigh/simlow, same/short/long, and adjacent/eightish/sixteenish for MDTO, MDTS, and MDTT, respectively. This is based on Vanderlip's (2025) exclusion of PrimacyRrecency for MDTT and applied consistently across domains
6. MDTO and MDTS have retained all previous derived measures
7. MDTT additionally has Temporal Discrimination Indices (adjacent, short, long, and overall) which discriminate against PrimacyRecency performance as a reference

*****************************************************************
'''

from __future__ import division
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from numpy import trapezoid
import sys

mapping = {
    "mdto": {
        "d_nearest":  "target",
        "cdNearer":   "simhigh",
        "cdFurther":  "simlow",
        "cdFurthest": "foil"
    },
    "mdts": {
        "cdNearest":  "same",
        "cdNearer":   "short",
        "cdFurther":  "long",
        "cdFurthest": "corner"
    },
    "mdtt": {
        "cdNearest" : "adjacent",
        "cdNearer":   "eightish",
        "cdFurther":  "sixteenish",
        "cdFurthest": "primacyrecency"
    },
}


def create_pandas_dataframe() -> dict:
    task_types_lc = ["mdto", "mdts", "mdtt"]

    df = pd.DataFrame(columns=["id_participant", "id_session", "visit_date"])

    dict_keys = {}

    for task_type_lc in task_types_lc:
        cdNearest, cdNearer, cdFurther, cdFurthest = mapping[task_type_lc].values()

        dict_keys["%s_raw" % (task_type_lc)] = [
            "%s_condition_trials" % (task_type_lc),
            "%s_%s_correct_count" % (task_type_lc, cdNearest), "%s_%s_incorrect_count" % (task_type_lc, cdNearest), "%s_%s_response_count" % (task_type_lc, cdNearest), "%s_%s_correct_pct" % (task_type_lc, cdNearest), "%s_%s_incorrect_pct" % (task_type_lc, cdNearest),
            "%s_%s_correct_count" % (task_type_lc, cdNearer), "%s_%s_incorrect_count" % (task_type_lc, cdNearer), "%s_%s_response_count" % (task_type_lc, cdNearer), "%s_%s_correct_pct" % (task_type_lc, cdNearer), "%s_%s_incorrect_pct" % (task_type_lc, cdNearer),
            "%s_%s_correct_count" % (task_type_lc, cdFurther), "%s_%s_incorrect_count" % (task_type_lc, cdFurther), "%s_%s_response_count" % (task_type_lc, cdFurther), "%s_%s_correct_pct" % (task_type_lc, cdFurther), "%s_%s_incorrect_pct" % (task_type_lc, cdFurther),
            "%s_%s_correct_count" % (task_type_lc, cdFurthest), "%s_%s_incorrect_count" % (task_type_lc, cdFurthest), "%s_%s_response_count" % (task_type_lc, cdFurthest), "%s_%s_correct_pct" % (task_type_lc, cdFurthest), "%s_%s_incorrect_pct" % (task_type_lc, cdFurthest)
        ]

        df[dict_keys["%s_raw" % (task_type_lc)]] = None

        match task_type_lc:
            case 'mdto' | 'mdts':
                dict_keys["%s_derived" % (task_type_lc)] = [
                    "%s_accuracy_overall" % (task_type_lc), "%s_accuracy_similar" % (task_type_lc),
                    "%s_recognition" % (task_type_lc),
                    "%s_ldi_combined" % (task_type_lc), "%s_ldi_slope" % (task_type_lc), "%s_ldi_high" % (task_type_lc), "%s_ldi_low" % (task_type_lc),
                    "%s_dprime_overall" % (task_type_lc), "%s_dprime_high" % (task_type_lc), "%s_dprime_low" % (task_type_lc),
                    "%s_auc_nearestfurthest" % (task_type_lc), "%s_auc_ldi" % (task_type_lc)
                ]

                df[dict_keys["%s_derived" % (task_type_lc)]] = None

            case 'mdtt':
                dict_keys["%s_derived" % (task_type_lc)] = [
                    "%s_accuracy_overall" % (task_type_lc), "%s_accuracy_similar" % (task_type_lc),
                    "%s_tdi_combined" % (task_type_lc), "%s_tdi_adjacent" % (task_type_lc), "%s_tdi_short" % (task_type_lc), "%s_tdi_long" % (task_type_lc)
                ]

                df[dict_keys["%s_derived" % (task_type_lc)]] = None

    return df, dict_keys


def get_data_after_colon(line: str) -> int:
    colon_idx = line.find(":")
    return int(line[colon_idx+1:].strip())


def Generate_Report(parent_folder)-> pd.DataFrame:
    error_log = "Data Parsed, Error Occured During, notes" + "\n"
    # to see what each block of code is doing examine error_level statement
    error_level = ""
    try:
        # Create pandas dataframe for each task type
        error_level = "Creating pandas dataframe,"
        df, dict_keys = create_pandas_dataframe()

        error_level = "Extracting data from all relevant log files,"
        input_dir = Path(parent_folder).iterdir()

        for f in input_dir:

            error_level = "Checking that log file is viable,"

            if not ((f.name.endswith("log.txt") or f.name.endswith("log_tau.txt")) and "old" not in f.name):
                continue

            text = f.read_text()
            lines = text.split("\n")

            if not any(line.startswith("Scores:") for line in lines):
                continue

            print(f"Writing data from {f.name}")

            subject_num = int(f.name.split("_")[0])
            session_num = int(f.name.split("_")[1])
            # uncomment if we want to add support for tau log files
            # if f.name.endswith("log_tau.txt"):
                # subject_num = f'{subject_num}_tau'
            task_type = f.name.split("_")[2]
            match task_type:
                case "MDTO":
                    task_type_lc = 'mdto'
                case "MDTS":
                    task_type_lc = 'mdts'
                case "MDTT":
                    task_type_lc = 'mdtt'

            date_idx = lines[0].find("on ")
            visit_date = str(lines[0][date_idx+3:].strip())

            trial_condition = 0

            error_level = f"Parsing data from log file of task type {task_type},"
            idx = 0
            while idx < len(lines):
                if task_type == "MDTT" and lines[idx].startswith("Blocks ran"):
                    trial_condition = get_data_after_colon(lines[idx])
                    break

                elif ((task_type == "MDTO" or task_type == "MDTS") and lines[idx].startswith("Trials/Condition")):
                    trial_condition = get_data_after_colon(lines[idx])
                    break
                idx += 1


            row_dict = {"id_participant": subject_num, "id_session": session_num, "visit_date": visit_date}
            score_idx = lines.index("Scores:")
            cdNearest, cdNearer, cdFurther, cdFurthest = mapping[task_type_lc].values()

            row_idx = 2
            for cd in [cdNearest, cdNearer, cdFurther, cdFurthest]:
                row_dict["%s_%s_correct_count" % (task_type_lc, cd)] = get_data_after_colon(lines[score_idx+row_idx])
                row_dict["%s_%s_incorrect_count" % (task_type_lc,  cd)] = get_data_after_colon(lines[score_idx+row_idx+1])
                row_dict["%s_%s_response_count" % (task_type_lc, cd)] = get_data_after_colon(lines[score_idx+row_idx+2])
                row_idx += 3

                if row_dict["%s_%s_response_count" % (task_type_lc, cd)] != 0:
                    row_dict["%s_%s_correct_pct" % (task_type_lc, cd)] = row_dict["%s_%s_correct_count" % (task_type_lc, cd)] / row_dict["%s_%s_response_count" % (task_type_lc, cd)]
                    row_dict["%s_%s_incorrect_pct" % (task_type_lc, cd)] = row_dict["%s_%s_incorrect_count" % (task_type_lc, cd)] / row_dict["%s_%s_response_count" % (task_type_lc, cd)]
                else:
                    row_dict["%s_%s_correct_pct" % (task_type_lc, cd)] = 0
                    row_dict["%s_%s_incorrect_pct" % (task_type_lc, cd)] = 0

            mnemonic_responses = sum([row_dict["%s_%s_response_count" % (task_type_lc, cdNearest)], row_dict["%s_%s_response_count" % (task_type_lc, cdNearer)], row_dict["%s_%s_response_count" % (task_type_lc, cdFurther)]])

            # If there are no responses to the first 3 image types, there is not enough information to derive variables, so skip that.
            if mnemonic_responses != 0:
                error_level = "Adding row to dataframe,"

                if task_type == "MDTO":
                    error_level = "Parsing data for MDTO derived rows,"

                    row_dict["mdto_condition_trials"] = trial_condition

                    row_dict["mdto_accuracy_overall"] = sum([row_dict['mdto_target_correct_count'], row_dict['mdto_simhigh_correct_count'], row_dict['mdto_simlow_correct_count'], row_dict['mdto_foil_correct_count']]) / sum([row_dict['mdto_target_response_count'], row_dict['mdto_simhigh_response_count'], row_dict['mdto_simlow_response_count'], row_dict['mdto_foil_response_count']])
                    row_dict["mdto_accuracy_similar"] = sum([row_dict['mdto_target_correct_count'], row_dict['mdto_simhigh_correct_count'], row_dict['mdto_simlow_correct_count']]) / sum([row_dict['mdto_target_response_count'], row_dict['mdto_simhigh_response_count'], row_dict['mdto_simlow_response_count']])

                    row_dict["mdto_recognition"] = row_dict['mdto_target_correct_pct'] - row_dict['mdto_foil_incorrect_pct'] 

                    row_dict["mdto_ldi_high"] = row_dict["mdto_simhigh_correct_pct"] - row_dict["mdto_target_incorrect_pct"]
                    row_dict["mdto_ldi_low"] = row_dict["mdto_simlow_correct_pct"] - row_dict["mdto_target_incorrect_pct"]
                    row_dict["mdto_ldi_slope"] = row_dict["mdto_ldi_low"] - row_dict["mdto_ldi_high"]
                    row_dict["mdto_ldi_combined"] = ((row_dict["mdto_ldi_high"] * row_dict["mdto_simhigh_response_count"]) + (row_dict["mdto_ldi_low"] * row_dict["mdto_simlow_response_count"])) / (row_dict["mdto_simhigh_response_count"] + row_dict["mdto_simlow_response_count"])

                    row_dict["mdto_dprime_overall"] = norm.ppf(min(max(row_dict["mdto_target_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdto_foil_incorrect_pct"], 0.0001), 0.9999))
                    row_dict["mdto_dprime_high"] = norm.ppf(min(max(row_dict["mdto_target_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdto_simhigh_incorrect_pct"], 0.0001), 0.9999))
                    row_dict["mdto_dprime_low"] = norm.ppf(min(max(row_dict["mdto_target_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdto_simlow_incorrect_pct"], 0.0001), 0.9999))

                    row_dict["mdto_auc_nearestfurthest"] = trapezoid([row_dict["mdto_target_correct_pct"], row_dict["mdto_foil_incorrect_pct"]], dx=1)
                    row_dict["mdto_auc_ldi"] = trapezoid([row_dict["mdto_ldi_low"], row_dict["mdto_ldi_high"]], dx=1)

                    # Check if both the participant ID and date exist in the dataframe; if so, advance; if not, append a new row
                    if (row_dict["id_participant"] in (df['id_participant'].values)) & (row_dict["visit_date"] in (df['visit_date'].values)):
                        # Check if those exist on the same line; if so, enter the new data on the existing line; if not, append a new row
                        if not df.loc[(df['id_participant'] == row_dict["id_participant"]) & (df['visit_date'] == row_dict["visit_date"])].empty:
                            df.loc[(df['id_participant'] == row_dict["id_participant"]) & (df['visit_date'] == row_dict["visit_date"]), row_dict.keys()] = row_dict.values()
                        else:
                            df.loc[len(df)] = row_dict
                    else:
                        df.loc[len(df)] = row_dict

                elif task_type == "MDTS":
                    error_level = "Parsing data for MDTS derived rows,"

                    row_dict["mdts_condition_trials"] = trial_condition

                    row_dict["mdts_accuracy_overall"] = sum([row_dict['mdts_same_correct_count'], row_dict['mdts_short_correct_count'], row_dict['mdts_long_correct_count'], row_dict['mdts_corner_correct_count']]) / sum([row_dict['mdts_same_response_count'], row_dict['mdts_short_response_count'], row_dict['mdts_long_response_count'], row_dict['mdts_corner_response_count']])
                    row_dict["mdts_accuracy_similar"] = sum([row_dict['mdts_same_correct_count'], row_dict['mdts_short_correct_count'], row_dict['mdts_long_correct_count']]) / sum([row_dict['mdts_same_response_count'], row_dict['mdts_short_response_count'], row_dict['mdts_long_response_count']])

                    row_dict["mdts_recognition"] = row_dict['mdts_same_correct_pct'] - row_dict['mdts_corner_incorrect_pct'] 

                    row_dict["mdts_ldi_high"] = row_dict["mdts_short_correct_pct"] - row_dict["mdts_same_incorrect_pct"]
                    row_dict["mdts_ldi_low"] = row_dict["mdts_long_correct_pct"] - row_dict["mdts_same_incorrect_pct"]
                    row_dict["mdts_ldi_slope"] = row_dict["mdts_ldi_low"] - row_dict["mdts_ldi_high"]
                    row_dict["mdts_ldi_combined"] = ((row_dict["mdts_ldi_high"] * row_dict["mdts_short_response_count"]) + (row_dict["mdts_ldi_low"] * row_dict["mdts_long_response_count"])) / (row_dict["mdts_short_response_count"] + row_dict["mdts_long_response_count"])

                    row_dict["mdts_dprime_overall"] = norm.ppf(min(max(row_dict["mdts_same_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdts_corner_incorrect_pct"], 0.0001), 0.9999))
                    row_dict["mdts_dprime_high"] = norm.ppf(min(max(row_dict["mdts_same_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdts_short_incorrect_pct"], 0.0001), 0.9999))
                    row_dict["mdts_dprime_low"] = norm.ppf(min(max(row_dict["mdts_same_correct_pct"], 0.0001), 0.9999)) - norm.ppf(min(max(row_dict["mdts_long_incorrect_pct"], 0.0001), 0.9999))

                    row_dict["mdts_auc_nearestfurthest"] = trapezoid([row_dict["mdts_same_correct_pct"], row_dict["mdts_corner_incorrect_pct"]], dx=1)
                    row_dict["mdts_auc_ldi"] = trapezoid([row_dict["mdts_ldi_low"], row_dict["mdts_ldi_high"]], dx=1)

                elif task_type == "MDTT":
                    error_level = "Parsing data for MDTT derived rows"
                    row_dict["mdtt_condition_trials"] = trial_condition

                    row_dict["mdtt_accuracy_overall"] = sum([row_dict['mdtt_adjacent_correct_count'], row_dict['mdtt_eightish_correct_count'], row_dict['mdtt_sixteenish_correct_count'], row_dict['mdtt_primacyrecency_correct_count']]) / sum([row_dict['mdtt_adjacent_response_count'], row_dict['mdtt_eightish_response_count'], row_dict['mdtt_sixteenish_response_count'], row_dict['mdtt_primacyrecency_response_count']])
                    row_dict["mdtt_accuracy_similar"] = sum([row_dict['mdtt_adjacent_correct_count'], row_dict['mdtt_eightish_correct_count'], row_dict['mdtt_sixteenish_correct_count']]) / sum([row_dict['mdtt_adjacent_response_count'], row_dict['mdtt_eightish_response_count'], row_dict['mdtt_sixteenish_response_count']])

                    row_dict["mdtt_tdi_adjacent"] = row_dict["mdtt_adjacent_correct_pct"] - row_dict["mdtt_primacyrecency_incorrect_pct"]
                    row_dict["mdtt_tdi_short"] = row_dict["mdtt_eightish_correct_pct"] - row_dict["mdtt_primacyrecency_incorrect_pct"]
                    row_dict["mdtt_tdi_long"] = row_dict["mdtt_sixteenish_correct_pct"] - row_dict["mdtt_primacyrecency_incorrect_pct"]
                    row_dict["mdtt_tdi_combined"] = ((row_dict["mdtt_tdi_adjacent"] * row_dict["mdtt_adjacent_response_count"]) + (row_dict["mdtt_tdi_short"] * row_dict["mdtt_eightish_response_count"]) + (row_dict["mdtt_tdi_long"] * row_dict["mdtt_sixteenish_response_count"])) / (row_dict["mdtt_adjacent_response_count"] + row_dict["mdtt_eightish_response_count"] + row_dict["mdtt_sixteenish_response_count"])

                # Check if both the participant ID and date exist in the dataframe; if so, advance; if not, append a new row
                if (row_dict["id_participant"] in (df['id_participant'].values)) & (row_dict["visit_date"] in (df['visit_date'].values)):
                    # Check if those exist on the same line; if so, enter the new data on the existing line; if not, append a new row
                    if not df.loc[(df['id_participant'] == row_dict["id_participant"]) & (df['visit_date'] == row_dict["visit_date"])].empty:
                        df.loc[(df['id_participant'] == row_dict["id_participant"]) & (df['visit_date'] == row_dict["visit_date"]), row_dict.keys()] = row_dict.values()
                    else:
                        df.loc[len(df)] = row_dict
                else:
                    df.loc[len(df)] = row_dict

            else:
                error_log = error_log + 'Rejected based on lack of respose,' + str(sys.exc_info()[1]) + f" File name: {f.name}" + ',\n'

        error_level = "Setting Output Directory,"

        df.sort_values(["id_participant", "id_session"])
        return df

    except Exception as e:
        #error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + f" File name: {f.name}" + ',\n'
        error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + '\n'
        print(error_log)