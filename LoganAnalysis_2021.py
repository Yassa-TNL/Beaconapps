'''
*****************************************************************#
Title                            Logan Analysis
Purpose                          Compiles dataset from
                                 multiple log files
                                 and last backup of each month
Creation Date                    08/27/2020
Created by                       John
Modified by                      Derek Vincent Taylor
Last Modified                    08/27/2020
*****************************************************************
In progress                     1. Add parse parameters for -s equals top folder and
                                -d equals output folder
                                2. Instead of values into excel sheet directly, parse
                                values into a pandas dataframe. Export pandas dataframe
                                into excel sheet.
                                3. Add recognition score hit rate - false alarm
                                Target-%Cor - Foil-%Inc
                                4. Calculate d' values handle 0 or 1 hit rate
                                or false alarm
                                5. d' for lures value d' equation
                                {Target-%Cor ,LureH-%Inc, LureL-%inc} see clack
                                see normsinv function equation
                                d'LureH = Z(HR of Target) -z(FA of LuresHigh)
                                6. Run script on cron job on server

'''

from __future__ import division

from datetime import datetime
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import numpy as np
from numpy import trapz
import sys

mapping = {
    "MDTO": {
        "cdHard": "Target",
        "cdHigh": "LureH",
        "cdLow": "LureL",
        "cdEasy": "Foil"
    },
    "MDTS": {
        "cdHard": "Same",
        "cdHigh": "Small Mv",
        "cdLow": "Large Mv",
        "cdEasy": "Corner Mv"
    },
    "MDTT": {
        "cdHard" : "Adj",
        "cdHigh": "Eight",
        "cdLow": "Sixteen",
        "cdEasy": "PR"
    },
}


def create_pandas_dataframes() -> dict:
    types = ["MDTO", "MDTS", "MDTT", "MDTO-LDI", "MDTS-LDI"]
    dfs = []

    for type in types:
        if type in ["MDTO", "MDTS", "MDTT"]:
            cdHard, cdHigh, cdLow, cdEasy = mapping[type].values()

            column_names = (
                "Subject", "Trials/Cond", "%s-Responses" % (cdHard), "%s-%%Cor" % (cdHard),
                "%s-%%Inc" % (cdHard), "%s-Responses" % (cdHigh), "%s-%%Cor" % (cdHigh), "%s-%%Inc" % (cdHigh),
                "%s-Responses" % (cdLow),
                "%s-%%Cor" % (cdLow), "%s-%%Inc" % (cdLow), "%s-Responses" % (cdEasy), "%s-%%Cor" % (cdEasy),
                "%s-%%Inc" % (cdEasy)
            )

            dfs.append(pd.DataFrame(columns=column_names))
        else:
            dfs.append(pd.DataFrame(columns=[
                "Subject", "d'", "LDI: High", "LDI: Combined", "Target-Foil_AUC", "LDI_slope",
                "Target-Foil_slope"
            ]))

    return dict(zip(types, dfs))


def get_data_after_colon(line: str) -> int:
    colon_idx = line.find(":")
    return int(line[colon_idx+1:].strip())


def get_proportion_data(line: str) -> float:
    return float(line[-4:])


def main():
    error_log = "Data Parsed, Error Occured During, notes" + "\n"
    # to see what each block of code is doing examine error_level statement
    error_level = ""
    try:
        error_level = 'Get Parameters, Declare Arg Parser,'
        parser = argparse.ArgumentParser()
        error_level = "Get Parameters, Add arguments to parser"
        parser.add_argument('-s', required=True, help="Top folder from which to read results")
        parser.add_argument('-d', required=True, help="Output folder for Results")
        error_level = "Get Parameters, Run Args Parser,"
        args = parser.parse_args()
        error_level = "Get Parameters, Declare Arg Parser,"

        # Create pandas dataframe for each task type
        error_log = "Creating pandas dataframes and logging data\n"
        error_level = "Creating pandas dataframes"
        dataframes = create_pandas_dataframes()

        # print(dataframes["MDTO"])

        error_level = "Extracting data from all relevant log files"
        input_dir = Path(args.s).iterdir()

        for f in input_dir:
            error_level = "Checking that log file is viable"
            if not (f.name.endswith("log.txt") and "old" not in f.name):
                continue

            text = f.read_text()
            lines = text.split("\n")

            if not any(line.startswith("Scores:") for line in lines):
                continue

            subject_num = int(f.name.split("_")[0])
            task_type = f.name.split("_")[1]
            trial_condition = 0

            error_level = f"Parsing data from log file of task type {task_type}"
            idx = 0
            while idx < len(lines):
                if task_type == "MDTT" and lines[idx].startswith("Blocks ran"):
                    trial_condition = get_data_after_colon(lines[idx])
                    break

                elif ((task_type == "MDTO" or task_type == "MDTS") and lines[idx].startswith("Trials/Condition")):
                    trial_condition = get_data_after_colon(lines[idx])
                    break
                idx += 1

            row_dict = {"Subject": subject_num, "Trials/Cond": trial_condition}
            score_idx = lines.index("Scores:")
            cdHard, cdHigh, cdLow, cdEasy = mapping[task_type].values()

            row_dict["%s-Responses" % cdHard] = get_data_after_colon(lines[score_idx+4])
            row_dict["%s-%%Cor" % cdHard] = get_proportion_data(lines[score_idx+15])
            row_dict["%s-%%Inc" % cdHard] = get_proportion_data(lines[score_idx + 16])
            row_dict["%s-Responses" % cdHigh] = get_data_after_colon(lines[score_idx + 7])
            row_dict["%s-%%Cor" % cdHigh] = get_proportion_data(lines[score_idx + 17])
            row_dict["%s-%%Inc" % cdHigh] = get_proportion_data(lines[score_idx + 18])
            row_dict["%s-Responses" % cdLow] = get_data_after_colon(lines[score_idx + 10])
            row_dict["%s-%%Cor" % cdLow] = get_proportion_data(lines[score_idx + 19])
            row_dict["%s-%%Inc" % cdLow] = get_proportion_data(lines[score_idx + 20])
            row_dict["%s-Responses" % cdEasy] = get_data_after_colon(lines[score_idx + 13])
            row_dict["%s-%%Cor" % cdEasy] = get_proportion_data(lines[score_idx + 21])
            row_dict["%s-%%Inc" % cdEasy] = get_proportion_data(lines[score_idx + 22])

            dataframes[task_type] = dataframes[task_type].append(row_dict, ignore_index = True)

            if task_type == "MDTS":
                error_level = "Parsing data for MDTS LDI dataframe"
                ldi_dict = {"Subject": subject_num}

                pTgtHit = row_dict["Same-%Cor"]
                pFoilFA = row_dict["Corner Mv-%Inc"]

                if pTgtHit == 1: pTgtHit = 0.9999999999
                if not pTgtHit: pTgtHit = 0.0000000001
                if pFoilFA == 1: pFoilFA = 0.9999999999
                if not pFoilFA: pFoilFA = 0.0000000001

                dPrime = norm.ppf(pTgtHit) - norm.ppf(pFoilFA)
                ldi_dict["d'"] = dPrime

                ldi_dict["LDI: High"] = row_dict["Small Mv-%Cor"] - row_dict["Same-%Inc"]
                ldi_dict["LDI: Low"] = row_dict["Large Mv-%Cor"] - row_dict["Same-%Inc"]
                ldi_dict["LDI: Combined"] = (ldi_dict["LDI: High"] + ldi_dict["LDI: Low"])/2
                yLDI = [ldi_dict["LDI: Low"], ldi_dict["LDI: High"]]
                ldi_dict["LDI_AUC"] = trapz(yLDI, dx=1)
                yTarFoil = [pTgtHit, pFoilFA]
                ldi_dict["Target-Foil_AUC"] = trapz(yTarFoil, dx=1)
                ldi_dict["LDI_slope"] = ldi_dict["LDI: Low"] - ldi_dict["LDI: High"]
                ldi_dict["Target-Foil_slope"] = pTgtHit - pFoilFA

                dataframes["MDTO-LDI"] = dataframes["MDTO-LDI"].append(ldi_dict, ignore_index=True)

            elif task_type == "MDTO":
                error_level = "Parsing data for MDTO LDI dataframe"
                ldi_dict = {"Subject": subject_num}

                pTgtHit = row_dict["Target-%Cor"]
                pFoilFA = row_dict["Foil-%Inc"]

                if pTgtHit == 1: pTgtHit = 0.9999999999
                if not pTgtHit: pTgtHit = 0.0000000001
                if pFoilFA == 1: pFoilFA = 0.9999999999
                if not pFoilFA: pFoilFA = 0.0000000001

                dPrime = norm.ppf(pTgtHit) - norm.ppf(pFoilFA)
                ldi_dict["d'"] = dPrime

                ldi_dict["LDI: High"] = row_dict["LureH-%Cor"] - row_dict["Target-%Inc"]
                ldi_dict["LDI: Low"] = row_dict["LureL-%Cor"] - row_dict["Target-%Inc"]
                ldi_dict["LDI: Combined"] = (ldi_dict["LDI: High"] + ldi_dict["LDI: Low"])/2
                yLDI = [ldi_dict["LDI: Low"], ldi_dict["LDI: High"]]
                ldi_dict["LDI_AUC"] = trapz(yLDI, dx=1)
                yTarFoil = [pTgtHit, pFoilFA]
                ldi_dict["Target-Foil_AUC"] = trapz(yTarFoil, dx=1)
                ldi_dict["LDI_slope"] = ldi_dict["LDI: Low"] - ldi_dict["LDI: High"]
                ldi_dict["Target-Foil_slope"] = pTgtHit - pFoilFA

                dataframes["MDTS-LDI"] = dataframes["MDTS-LDI"].append(ldi_dict, ignore_index=True)

        output_dir = str(args.d)
        error_level = "Writing pandas dataframes to excel sheet"

        datestamp = datetime.now()
        result_name = datestamp.strftime(f"{output_dir}BeaconAppResults_%Y_%m_%d.xlsx")
        print(f"result_name: {result_name}")

        for key, value in dataframes.items():
            value.to_excel(result_name, sheet_name=key)

    except Exception as e:
        error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + f"\n, File name: {f.name}"
        print(error_log)
        raise

if __name__ == '__main__': main()
