'''

***********************************************************
Title                            Class Assembly compiles
Purpose                          Table of Results from
                                 ClassifierResults.txt files
                                 output from DiagnosisClassifier.py
Creation Date                    08/08/2021
Created by                       Derek V. Taylor
#*****************************************************************#
:param source: string; path to parent folder for Classifier Results files
:param dest: string; path to output folder were ClassifierSummary csv will be deposited
:Return ClassifierSummary_Y_M_d.csv: file; contains compiled results labeled by year mongh and day in file name

'''
from datetime import datetime
import os
from os import path
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import glob
import shutil
import argparse
import numpy as np

def main():
    error_log = "Data Parsed, Error Occured During, Record, Notes" + '\n'
    error_level = ""
    try:
        error_level = "Get Parameters, Declare Arg Parser,"
        parser = argparse.ArgumentParser()
        error_level = "Get Parameters, Add arguments to parser,"
        parser.add_argument('-s', required=True, help="Top folder from which to read results")
        parser.add_argument('-d', required=True, help="Output folder for Results")
        error_level = "Get Parameters, Run Args Parser,"
        args = parser.parse_args()
        error_level = "Get Parameters, Declare Arg Parser,"
        source = args.s
        rfound = 0
        dest = args.d
        if path.exists(source):
            spath= join(source,"*ClassifierResults*")
            rpt_headers = 'Folder,Model,cross_v_numb,predict_var_used,out_var_used,classes_in_outcome,' \
                          'mean_chance_prob,std_chance_prob,ci_lb,ci_ub,' \
                          'accuracy,accuracy_sd,accuracy_lb,accuracy_ub,' \
                          'precision,precision_sd,precision_lb,precision_ub,' \
                          'recall,recall_sd,recall_lb,recall_ub,' \
                          'fbeta,fbeta_sd,fbeta_lb,fbeta_ub,' \
                          'auc,auc_sd,auc_lb,auc_ub'
            for x in os.walk(source):
                try:
                    spath= join(x[0],"*ClassifierResults*")
                    for F in glob.glob(spath):
                        if not "@eaDir" in F:
                            print("Parsing",F)
                            folder = get_folder_name(F)
                            with open(F,'r') as rfile:
                                results = rfile.readlines()
                            error_level = "Iterating through files, Interating through files,"
                            for l_numb,l in enumerate(results, 1):
                                try:
                                    error_level = "Parsing header, read line,"
                                    l = l.strip()
                                    error_level = "Parsing Header Data, Get Cross validation number,"
                                    if "Cross validation number of stratified splits" in l:
                                        cross_v_numb = l.split(':')[1]
                                        cross_v_numb = float(cross_v_numb)
                                    elif "Predictor Variables Used" in l:
                                        error_level = "Parsing Header Data, Get predictor var used,"
                                        predict_var_used =  l.split(':')[1]
                                    elif "Outcome Variable Used" in l:
                                        error_level = "Parsing Header Data, Get out var used,"
                                        out_var_used = l.split(':')[1]
                                    elif "Classes in Outcome Variable" in l:
                                        error_level = "Parsing Header Data, Get classes in outcome,"
                                        classes_in_outcome = l.split(':')[1]
                                    elif "mean chance prob" in l:
                                        error_level = "Parsing Header Data, Get mean chance prob,"
                                        mean_chance_prob = l.split('=')[1].strip()
                                        mean_chance_prob = mean_chance_prob.replace(',','')
                                        mean_chance_prob = float(mean_chance_prob)
                                    elif "std chance prob" in l:
                                        error_level = "Parsing Header Data, Get standard prob chance,"
                                        std_chance_prob = l.split('=')[1].strip()
                                        std_chance_prob = float( std_chance_prob)
                                    elif "-----------------------" in l:
                                        rfound +=1
                                        if rfound > 1:
                                            error_level = "Parsing header, read line,"
                                            conf_inc = get_conf_inc( std_chance_prob,cross_v_numb)
                                            ci_lb = mean_chance_prob - conf_inc
                                            ci_ub = mean_chance_prob + conf_inc
                                            rpt_dict = get_rpt_dictionary(rpt_headers)
                                            rpt_dict['Folder'] = folder
                                            rpt_dict['Model'] = Model
                                            rpt_dict['cross_v_numb'] = cross_v_numb
                                            rpt_dict['predict_var_used'] = predict_var_used
                                            rpt_dict['ci_lb'] = ci_lb
                                            rpt_dict['ci_ub'] = ci_ub
                                            rpt_dict['out_var_used'] = out_var_used
                                            rpt_dict['classes_in_outcome'] = classes_in_outcome
                                            rpt_dict['mean_chance_prob'] = mean_chance_prob
                                            rpt_dict['std_chance_prob'] = std_chance_prob
                                            rpt_dict['accuracy'] = accuracy
                                            rpt_dict['accuracy_sd'] = accuracy_sd
                                            rpt_dict['precision'] = precision
                                            rpt_dict['precision_sd'] = precision_sd
                                            rpt_dict['recall'] = recall
                                            rpt_dict['recall_sd'] = recall_sd
                                            rpt_dict['fbeta'] = fbeta
                                            rpt_dict['fbeta_sd'] = fbeta_sd
                                            rpt_dict['auc'] = auc
                                            rpt_dict['auc_sd'] = auc_sd
                                            rpt_dict['accuracy_lb'] = accuracy_lb
                                            rpt_dict['accuracy_ub'] = accuracy
                                            rpt_dict['auc_lb'] = auc_lb
                                            rpt_dict['auc_ub'] = auc_ub
                                            rpt_dict['precision_lb'] = precision_lb
                                            rpt_dict['precision_ub'] = precision_ub
                                            rpt_dict['fbeta_lb'] = fbeta_lb
                                            rpt_dict['fbeta_ub'] = fbeta_ub
                                            rpt_dict['recall_lb'] = recall_lb
                                            rpt_dict['recall_ub'] = recall_ub
                                            df_rpt_row = pd.DataFrame(rpt_dict, index=[0])
                                            if rfound > 2:
                                                df_rpt = df_rpt.append(df_rpt_row)
                                            else:
                                                df_rpt = df_rpt_row
                                        error_level = "Parsing results, Getting result type,"
                                        Model = results[l_numb-2].split(':')[0].replace('results','')
                                    elif 'accuracy=' in l:
                                        error_level = "Parsing results, Getting accuracy,"
                                        ac = l.split('=')[1]
                                        accuracy = (ac.split('+/-')[0]).strip()
                                        accuracy_sd = (ac.split('+/-')[1]).strip()
                                        accuracy = float(accuracy)
                                        accuracy_sd = float(accuracy_sd)
                                        accuracy_ci = get_conf_inc(accuracy_sd,cross_v_numb)
                                        accuracy_lb = accuracy - accuracy_ci
                                        accuracy_ub = accuracy + accuracy_ci
                                    elif 'precision=' in l:
                                        error_level = "Parsing results, Getting  precision,"
                                        pec = l.split('=')[1]
                                        precision = (pec.split('+/-')[0]).strip()
                                        precision = float(precision)
                                        precision_sd = (pec.split('+/-')[1]).strip()
                                        precision = float(precision)
                                        precision_sd = float(precision_sd)
                                        precision_ci = get_conf_inc(precision_sd,cross_v_numb)
                                        precision_lb = precision - precision_ci
                                        precision_ub = precision + precision_ci
                                    elif 'recall=' in l:
                                        error_level = "Parsing results, Getting recall,"
                                        rec = l.split('=')[1]
                                        recall = (rec.split('+/-')[0]).strip()
                                        recall_sd = (rec.split('+/-')[1]).strip()
                                        recall = float(recall)
                                        recall_sd = float(recall_sd)
                                        recall_ci = get_conf_inc(recall_sd,cross_v_numb)
                                        recall_lb = recall - recall_ci
                                        recall_ub = recall + recall_ci
                                    elif 'fbeta=' in l:
                                        error_level = "Parsing results, Getting fbetay,"
                                        fb = l.split('=')[1]
                                        fbeta = (fb.split('+/-')[0]).strip()
                                        fbeta_sd = (fb.split('+/-')[1]).strip()
                                        fbeta = float(fbeta)
                                        fbeta_sd = float(fbeta_sd)
                                        fbeta_ci = get_conf_inc(fbeta_sd,cross_v_numb)
                                        fbeta_lb = fbeta - fbeta_ci
                                        fbeta_ub = fbeta + fbeta_ci
                                    elif 'auc=' in l:
                                        error_level = "Parsing results, Getting AUC,"
                                        ac = l.split('=')[1]
                                        auc = (ac.split('+/-')[0]).strip()
                                        auc_sd = (ac.split('+/-')[1]).strip()
                                        auc = float(auc)
                                        auc_sd = float(auc_sd)
                                        auc_ci = get_conf_inc(auc_sd,cross_v_numb)
                                        auc_lb = auc - auc_ci
                                        auc_ub = auc + auc_ci

                                except:
                                    error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + '\n'
                except:
                    error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + ": " + F + '\n'
            error_level = "Parsing results, Initiating rpt_headers,"
            error_level = "Parsing results, Outputting Data,"
            data = df_rpt.to_csv(header=True, index=False, columns=rpt_headers.split(','))
            error_level = "Parsing results, Setting Data Stamp for file label,"
            datestamp = datetime.now()
            ofile =  datestamp.strftime("ClassifierSummary_%Y_%m_%d.csv")
            error_level = "Parsing results, Joining filename to path,"
            ofile = join(dest,ofile)
            error_level = "Parsing results, Writing File,"
            with open(ofile,'w') as fl:
                fl.write(data)
            print("Output sent to:",ofile)
        else:
            print(source,'Not a valid path')

    except:
        error_log = error_log + error_level + ',' + str(sys.exc_info()[1]) + '\n'
    print(error_log)


def get_rpt_dictionary(rpt_headers):
    try:
        rpt_dic = rpt_headers.replace(',', '= '' ,') + '='''
        rpt_dic = dict(x.split("=") for x in rpt_dic.split(','))
        return rpt_dic
    except:
        return None


def get_file_name(pname):
    parts = pname.split('/')
    end = len(parts)-1
    fname = parts[end]
    return fname


def get_folder_name(pname):
    sep = os.path.sep
    parts = pname.split(sep)
    end = len(parts)-2
    sfolder = parts[end]
    return sfolder


def get_conf_inc(std,n):
    try:
        inc = 1.96*(std/np.sqrt(n))
        return inc
    except:
        return -1

if __name__ == '__main__':
    main()
