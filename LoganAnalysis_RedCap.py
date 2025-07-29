'''
*****************************************************************#
Title                            Logan Analysis
Purpose                          Compiles dataset from
                                 multiple log files
                                 and last backup of each month
Creation Date                    08/27/2020
Created by                       John
Modified by                      Derek Vincent Taylor, Steve Flores, Jason R. Bock
Last Modified                    07/16/2025
last modified by                 Derek Vincent Taylor:
*****************************************************************


Changed lines 207, 252, and 295 to use weighted mean rather than simple mean for LDI: Combined calculation, to account for unequal non-response to low- and high-lure items.
'''

from __future__ import division

from datetime import datetime
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from numpy import trapz
import sys
import requests
from requests import post

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

redcap_names = {
    "MDTO":
         {'Trials/Cond' : 'mdto_trials_cond',
        'Target-Responses' : 'mdto_target_responses',
        'Target-%Cor' : 'mdto_target_pct_cor',
        'Target-%Inc' : 'mdto_target_pct_inc',
        'LureH-Responses' : 'mdto_lureh_responses',
        'LureH-%Cor' : 'mdto_lureh_pct_cor',
        'LureH-%Inc' : 'mdto_lureh_pct_inc',
        'LureL-Responses' : 'mdto_lurel_responses',
        'LureL-%Cor' : 'mdto_lurel_pct_cor',
        'LureL-%Inc' : 'mdto_lurel_pct_inc',
        'Foil-Responses' : 'mdto_foil_responses',
        'Foil-%Cor' : 'mdto_foil_pct_cor',
        'Foil-%Inc' : 'mdto_foil_pct_inc'
          },
    "MDTS":
        {'Trials/Cond' : 'mdts_trials_cond',
        'Same-Responses' : 'mdts_same_responses',
        'Same-%Cor' : 'mdts_same_pct_cor',
        'Same-%Inc' : 'mdts_same_pct_inc',
        'Small Mv-Responses' : 'mdts_small_mv_responses',
        'Small Mv-%Cor' : 'mdts_small_mv_pct_cor',
        'Small Mv-%Inc' : 'mdts_small_mv_pct_inc',
        'Large Mv-Responses' : 'mdts_large_mv_responses',
        'Large Mv-%Cor' : 'mdts_large_mv_pct_cor',
        'Large Mv-%Inc' : 'mdts_large_mv_pct_inc',
        'Corner Mv-Responses' : 'mdts_corner_mv_responses',
        'Corner Mv-%Cor' : 'mdts_corner_mv_pct_cor',
        'Corner Mv-%Inc' : 'mdts_corner_mv_pct_inc',
        },
    "MDTT" :
        {'Trials/Cond' : 'mdtt_trials_cond',
        'Adj-Responses' : 'mdtt_adj_responses',
        'Adj-%Cor' : 'mdtt_adj_pcnt_cor',
        'Adj-%Inc' : 'mdtt_adj_pct_inc',
        'Eight-Responses' : 'mdtt_eight_responses',
        'Eight-%Cor' : 'mdtt_eight_pct_cor',
        'Eight-%Inc' : 'mdtt_eight_inc',
        'Sixteen-Responses' : 'mdtt_sixteen_responses',
        'Sixteen-%Cor' : 'mdtt_sixteen_pct_cor',
        'Sixteen-%Inc' : 'mdtt_sixteen_pct_inc',
        'PR-Responses' : 'mdtt_pr_responses',
        'PR-%Cor' : 'mdtt_pr_pct_cor',
        'PR-%Inc' : 'mdtt_pr_pct_inc',
        },
'MDTO-LDI' :
        {"Recognition" : "mdto_recognition",
        "d'" : "mdto_ldi_d_prime",
        "LDI: High" : "mdto_ldi_high",
        "LDI: Combined" : "mdto_ldi_combined",
        "Target-Foil_AUC" : "mdto_ldi_target_foil_auc",
        "LDI_slope" : "mdto_ldi_slope",
        "Target-Foil_slope" : "mdto_ldi_target_foil_slope",
        "d'LureH" : "mdto_ldi_low",
        "d'LureL" : "mdto_ldi_auc",
        "LDI: Low" : "mdto_d_prime_lure_h",
        "LDI_AUC" : "mdto_d_prime_lure_l"
        },
'MDTS-LDI' :
        {"Recognition" : "mdts_recognition",
        "d'" : "mdts_d_prime",
        "LDI: High" : "mdts_ldi_high",
        "LDI: Combined" : "mdts_ldi_combined",
        "Target-Foil_AUC" : "mdts_target_foil_auc",
        "LDI_slope" : "mdts_ldi_slope",
        "Target-Foil_slope" : "mdts_target_foil_slope",
        "d'LureH" : "mdts_ldi_low",
        "d'LureL" : "mdts_ldi_auc",
        "LDI: Low" : "mdts_d_prime_lure_h",
        "LDI_AUC" : "mdts_d_prime_lure_l"
         },
"MDTT-LDI" :
        {"Recognition": "mdtt_recognition",
        "d'": "mdtt_ldi_d_prime",
        "LDI: High": "mdtt_ldi_high",
        "LDI: Combined": "mdtt_ldi_combined",
        "Target-Foil_AUC": "mdtt_target_foil_auc",
        "LDI_slope": "mdtt_ldi_slope",
        "Target-Foil_slope": "mdtt_target_foil_slope",
        "d'LureH": "mdtt_ldi_low",
        "d'LureL": "mdtt_ldi_auc",
        "LDI: Low": "mdtt_d_prime_lure_h",
        "LDI_AUC": "mdtt_d_prime_lure_l"
        }
}

redcap_headers =  {
    "MDTO" : 'record_id,redcap_event_name,mdto_trials_cond,mdto_target_responses,mdto_target_pct_cor,'
             'mdto_target_pct_inc,mdto_lureh_responses,mdto_lureh_pct_cor,mdto_lureh_pct_inc,mdto_lurel_responses,'
             'mdto_lurel_pct_cor,mdto_lurel_pct_inc,mdto_foil_responses,mdto_foil_pct_cor,mdto_foil_pct_inc,'
             'mdto_foil_pct_inc',
    "MDTS" : 'record_id,redcap_event_name,mdts_trials_cond,mdts_same_responses,mdts_same_pct_cor,mdts_same_pct_inc,'
             'mdts_small_mv_responses,mdts_small_mv_pct_cor,mdts_small_mv_pct_inc,mdts_large_mv_responses,'
             'mdts_large_mv_pct_cor,mdts_large_mv_pct_inc,mdts_corner_mv_responses,mdts_corner_mv_pct_cor,'
             'mdts_corner_mv_pct_inc',
    "MDTT" : 'record_id,redcap_event_name,mdtt_trials_cond,mdtt_adj_responses,mdtt_adj_pcnt_cor,mdtt_adj_pct_inc,'
             'mdtt_eight_responses,mdtt_eight_pct_cor,mdtt_eight_inc,mdtt_sixteen_responses,mdtt_sixteen_pct_cor,'
             'mdtt_sixteen_pct_inc,mdtt_pr_responses,mdtt_pr_pct_cor,mdtt_pr_pct_inc',
    "MDTO-LDI" : 'record_id,redcap_event_name,mdto_recognition,mdto_ldi_d_prime,mdto_ldi_high,mdto_ldi_combined,mdto_ldi_target_foil_auc,'
                 'mdto_ldi_slope,mdto_ldi_target_foil_slope,mdto_ldi_low,mdto_ldi_auc,mdto_d_prime_lure_h,'
                 'mdto_d_prime_lure_l',
    "MDTS-LDI" : 'record_id,redcap_event_name,mdts_recognition,mdts_d_prime,mdts_ldi_high,mdts_ldi_combined,'
                 'mdts_target_foil_auc,mdts_ldi_slope,'
                 'mdts_target_foil_slope,mdts_ldi_low,mdts_ldi_auc,mdts_d_prime_lure_h,mdts_d_prime_lure_l',
    "MDTT-LDI" : 'record_id,redcap_event_name,mdtt_recognition,mdtt_ldi_d_prime,mdtt_ldi_high,mdtt_ldi_combined,'
                 'mdtt_target_foil_auc,mdtt_ldi_slope,mdtt_target_foil_slope,mdtt_ldi_low,mdtt_ldi_auc,'
                 'mdtt_d_prime_lure_h,mdtt_d_prime_lure_l'
}

def create_pandas_dataframes() -> dict:
    types = ["MDTO", "MDTS", "MDTT", "MDTO-LDI", "MDTS-LDI", "MDTT-LDI"]
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
                "Subject","Recognition", "d'", "LDI: High", "LDI: Combined", "Target-Foil_AUC", "LDI_slope",
                "Target-Foil_slope","d'LureH", "d'LureL", "LDI: Low", "LDI_AUC"
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
        parser.add_argument('-a', required=True, help="RedCap token")
        parser.add_argument('-r', required=True, help="ID report number")
        parser.add_argument('-e', required=True, help="ID report number")
        error_level = "Get Parameters, Run Args Parser,"
        args = parser.parse_args()
        error_level = "Get Parameters, Set TOKEN Value,"
        TOKEN = args.a
        report_id = args.r
        redcap_event_name = args.e
        error_level = 'Pull ID Report'
        URL = 'https://ci-redcap.hs.uci.edu/api/'
        payload = {'content': 'report',
                   'token': TOKEN,
                   'format': 'json',
                   'report_id': report_id,
                   'rawOrLabel': 'raw',
                   'rawOrLabelHeaders': 'raw',
                   'exportCheckboxLabel': 'false',
                   'returnFormat': 'json'}
        response = post(URL, data=payload)
        id_list = pd.read_json(response.text)

        # Create pandas dataframe for each task type
        error_level = "Creating pandas dataframes,"
        dataframes = create_pandas_dataframes()

        # print(dataframes["MDTO"])

        error_level = "Extracting data from all relevant log files,"
        input_dir = Path(args.s).iterdir()

        for f in input_dir:
            error_level = "Checking that log file is viable,"
            # if we wanted to add support to tau log files, we would need to add a check here
            if not ((f.name.endswith("log.txt") or f.name.endswith("log_tau.txt")) and "old" not in f.name):
                continue

            text = f.read_text()
            lines = text.split("\n")

            if not any(line.startswith("Scores:") for line in lines):
                continue

            subject_num = int(f.name.split("_")[0])
            # uncomment if we want to add support for tau log files
            # if f.name.endswith("log_tau.txt"):
                # subject_num = f'{subject_num}_tau'
            task_type = f.name.split("_")[1]
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

            if row_dict["%s-Responses" % cdHard] != 0:
                error_level = "Adding row to dataframe,"

                dataframes[task_type].loc[len(dataframes[task_type])] = row_dict
                # dataframes[task_type] = dataframes[task_type].append(row_dict, ignore_index = True)
                if task_type == "MDTS":
                    error_level = "Parsing data for MDTS LDI dataframe,"
                    ldi_dict = {"Subject": subject_num}

                    pTgtHit = row_dict["Same-%Cor"]
                    pFoilFA = row_dict["Corner Mv-%Inc"]
                    pLureHFA = row_dict['Small Mv-%Inc']
                    pLureLFA = row_dict['Large Mv-%Inc']

                    if pTgtHit == 1: pTgtHit = 0.9999999999
                    if not pTgtHit: pTgtHit = 0.0000000001
                    if pFoilFA == 1: pFoilFA = 0.9999999999
                    if not pFoilFA: pFoilFA = 0.0000000001
                    if pLureLFA == 1: pLureLFA = 0.999999
                    if not pLureLFA: pLureLFA = 0.000001
                    if pLureHFA == 1: pLureHFA = 0.999999
                    if not pLureHFA: pLureHFA = 0.000001
                    dprime_lure_high = norm.ppf(pTgtHit) - norm.ppf(pLureHFA)
                    ldi_dict["d'LureH"] = dprime_lure_high
                    dprime_lure_low = norm.ppf(pTgtHit) - norm.ppf(pLureLFA)
                    ldi_dict["d'LureL"] = dprime_lure_low


                    dPrime = norm.ppf(pTgtHit) - norm.ppf(pFoilFA)
                    # added recognition score
                    ldi_dict["Recognition"] = row_dict['Same-%Cor'] - row_dict['Corner Mv-%Inc']
                    ldi_dict["d'"] = dPrime
                    ldi_dict["LDI: High"] = row_dict["Small Mv-%Cor"] - row_dict["Same-%Inc"]
                    ldi_dict["LDI: Low"] = row_dict["Large Mv-%Cor"] - row_dict["Same-%Inc"]
                    ldi_dict["LDI: Combined"] = ((ldi_dict["LDI: High"] * row_dict["Small Mv-Responses"]) + (ldi_dict["LDI: Low"] * row_dict["Large Mv-Responses"])) / (row_dict["Small Mv-Responses"] + row_dict["Large Mv-Responses"])
                    yLDI = [ldi_dict["LDI: Low"], ldi_dict["LDI: High"]]
                    ldi_dict["LDI_AUC"] = trapz(yLDI, dx=1)
                    yTarFoil = [pTgtHit, pFoilFA]
                    ldi_dict["Target-Foil_AUC"] = trapz(yTarFoil, dx=1)
                    ldi_dict["LDI_slope"] = ldi_dict["LDI: Low"] - ldi_dict["LDI: High"]
                    ldi_dict["Target-Foil_slope"] = pTgtHit - pFoilFA

                    # dataframes["MDTS-LDI"] = dataframes["MDTS-LDI"].append(ldi_dict, ignore_index=True)
                    dataframes["MDTS-LDI"].loc[len(dataframes["MDTS-LDI"])] = ldi_dict
                    dataframes["MDTS-LDI"].sort_values(by =['Subject'], inplace=True )

                elif task_type == "MDTO":
                    error_level = "Parsing data for MDTO LDI dataframe,"
                    ldi_dict = {"Subject": subject_num}

                    pTgtHit = row_dict["Target-%Cor"]
                    pFoilFA = row_dict["Foil-%Inc"]
                    pLureHFA = row_dict['LureH-%Inc']
                    pLureLFA = row_dict['LureL-%Inc']
                    if pTgtHit == 1: pTgtHit = 0.999999
                    if not pTgtHit: pTgtHit = 0.000001
                    if pFoilFA == 1: pFoilFA = 0.999999
                    if not pFoilFA: pFoilFA = 0.000001
                    if pLureLFA == 1: pLureLFA = 0.999999
                    if not pLureLFA: pLureLFA = 0.000001
                    if pLureHFA == 1: pLureHFA = 0.999999
                    if not pLureHFA: pLureHFA = 0.000001

                    # added recognition score
                    ldi_dict["Recognition"] = row_dict['Target-%Cor'] - row_dict['Foil-%Inc']
                    dPrime = norm.ppf(pTgtHit) - norm.ppf(pFoilFA)
                    ldi_dict["d'"] = dPrime
                    dprime_lure_high = norm.ppf(pTgtHit) - norm.ppf(pLureHFA)
                    ldi_dict["d'LureH"] = dprime_lure_high
                    dprime_lure_low = norm.ppf(pTgtHit) - norm.ppf(pLureLFA)
                    ldi_dict["d'LureL"] = dprime_lure_low
                    '''
                    5. d' for lures value d' equation
                                {Target-%Cor ,LureH-%Inc, LureL-%inc} see clack
                                see normsinv function equation
                    '''

                    ldi_dict["LDI: High"] = row_dict["LureH-%Cor"] - row_dict["Target-%Inc"]
                    ldi_dict["LDI: Low"] = row_dict["LureL-%Cor"] - row_dict["Target-%Inc"]
                    ldi_dict["LDI: Combined"] = ((ldi_dict["LDI: High"] * row_dict["LureH-Responses"]) + (ldi_dict["LDI: Low"] * row_dict["LureL-Responses"])) / (row_dict["LureH-Responses"] + row_dict["LureL-Responses"])
                    yLDI = [ldi_dict["LDI: Low"], ldi_dict["LDI: High"]]
                    ldi_dict["LDI_AUC"] = trapz(yLDI, dx=1)
                    yTarFoil = [pTgtHit, pFoilFA]
                    ldi_dict["Target-Foil_AUC"] = trapz(yTarFoil, dx=1)
                    ldi_dict["LDI_slope"] = ldi_dict["LDI: Low"] - ldi_dict["LDI: High"]
                    ldi_dict["Target-Foil_slope"] = pTgtHit - pFoilFA
                    #ldi_dict["dprime-high-lure"] =
                    dataframes["MDTO-LDI"].loc[len(dataframes["MDTO-LDI"])] = ldi_dict
                    # dataframes["MDTO-LDI"] = dataframes["MDTO-LDI"].append(ldi_dict, ignore_index=True)
                elif task_type == "MDTT":
                    error_level = "Parsing data for MDTT LDI dataframe"
                    ldi_dict = {"Subject": subject_num}

                    error_level = "Defining hits and false alarms"
                    pTgtHit = row_dict["Adj-%Cor"]
                    pFoilFA = row_dict["PR-%Inc"]
                    pLureHFA = row_dict['Eight-%Inc']
                    pLureLFA = row_dict['Sixteen-%Inc']

                    error_level = "Setting default values"
                    if pTgtHit == 1: pTgtHit = 0.999999
                    if not pTgtHit: pTgtHit = 0.000001
                    if pFoilFA == 1: pFoilFA = 0.999999
                    if not pFoilFA: pFoilFA = 0.000001
                    if pLureLFA == 1: pLureLFA = 0.999999
                    if not pLureLFA: pLureLFA = 0.000001
                    if pLureHFA == 1: pLureHFA = 0.999999
                    if not pLureHFA: pLureHFA = 0.000001

                    ldi_dict["Recognition"] = row_dict['Adj-%Cor'] - row_dict['PR-%Inc']
                    error_level = "Computing d'prime"
                    dPrime = norm.ppf(pTgtHit) - norm.ppf(pFoilFA)
                    ldi_dict["d'"] = dPrime
                    dprime_lure_high = norm.ppf(pTgtHit) - norm.ppf(pLureHFA)
                    ldi_dict["d'LureH"] = dprime_lure_high
                    dprime_lure_low = norm.ppf(pTgtHit) - norm.ppf(pLureLFA)
                    ldi_dict["d'LureL"] = dprime_lure_low

                    error_level = "Computing  LDI"
                    ldi_dict["LDI: High"] = row_dict["Eight-%Cor"] - row_dict["Adj-%Inc"]
                    ldi_dict["LDI: Low"] = row_dict["Sixteen-%Cor"] - row_dict["Adj-%Inc"]
                    ldi_dict["LDI: Combined"] = ((ldi_dict["LDI: High"] * row_dict["Eight-Responses"]) + (ldi_dict["LDI: Low"] * row_dict["Sixteen-Responses"])) / (row_dict["Eight-Responses"] + row_dict["Sixteen-Responses"])
                    yLDI = [ldi_dict["LDI: Low"], ldi_dict["LDI: High"]]
                    ldi_dict["LDI_AUC"] = trapz(yLDI, dx=1)
                    yTarFoil = [pTgtHit, pFoilFA]
                    ldi_dict["Target-Foil_AUC"] = trapz(yTarFoil, dx=1)
                    ldi_dict["LDI_slope"] = ldi_dict["LDI: Low"] - ldi_dict["LDI: High"]
                    ldi_dict["Target-Foil_slope"] = pTgtHit - pFoilFA
                    #ldi_dict["dprime-high-lure"] =
                    error_level = "Writing Data Frame"
                    dataframes["MDTT-LDI"].loc[len(dataframes["MDTT-LDI"])] = ldi_dict
                    # dataframes["MDTT-LDI"] = dataframes["MDTT-LDI"].append(ldi_dict, ignore_index=True)
            else:
                error_log = error_log + 'Rejected based on lack of respose,' + str(sys.exc_info()[1]) + f" File name: {f.name}" + ',\n'
        error_level = "Writing pandas dataframes to excel sheet,"

        for key, value in dataframes.items():
            value.sort_values(by=['Subject'], inplace=True)
            error_level = "Processing MDT data, renaming variables"
            df_data = value.rename(columns=redcap_names[key])
            error_level = "Processing MDT data,create sub_id field"
            df_data.loc[:,'sub_id'] = df_data['Subject']
            error_level = "Processing MDT data,inner merge to ID list"
            df_merged = pd.merge(df_data, id_list, on = ['sub_id'])
            error_level = "Processing MDT data,set redcap_event_name"
            df_merged.loc[:,'redcap_event_name'] = args.e
            error_level = "Processing MDT data,export data set"
            upload_data = df_merged.to_csv(columns=redcap_headers[key].split(","), header=True, index=False)
            try:
                DATA = upload_data
                payload = {'token': TOKEN, 'format': 'csv', 'content': 'record', 'data': DATA}
                response = post(URL, data=payload)
                # print(DATA)
                print(response.status_code, response.text)
            except:
                error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + '\n'
    except Exception as e:
        #error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + f" File name: {f.name}" + ',\n'
        error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + '\n'
    print(error_log)
if __name__ == '__main__': main()