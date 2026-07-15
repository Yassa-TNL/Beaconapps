from csv import Error
from io import StringIO

from LoganAnalysis_2026_Session import Generate_Report
from argparse import ArgumentParser
import pandas as pd
from requests import post

completion_variables = {
    "MDTO": 'mdto_complete',
    "MDTS": "mdts_complete",
    "MDTT": 'mdtt_complete',
    "MDTO-LDI": 'mdtoldi_complete',
    "MDTS-LDI": 'mdtsldi_complete',
    "MDTT-LDI": 'mdttldi_complete'
}
URL = 'https://ci-redcap.hs.uci.edu/api/'


def get_report(token, report_id) -> pd.DataFrame:
    payload = {'content': 'report',
               'token': token,
               'format': 'json',
               'report_id': report_id,
               'rawOrLabel': 'raw',
               'rawOrLabelHeaders': 'raw',
               'exportCheckboxLabel': 'false',
               'returnFormat': 'json'}
    response = post(URL, data=payload)
    id_list = pd.read_json(StringIO(response.text))
    return id_list


def upload_redcap(DATA, TOKEN):
    payload = {'token': TOKEN, 'format': 'csv',
               'content': 'record', 'data': DATA}
    response = post(URL, data=payload)
    print(response.status_code, response.text)


def normalize_report_date(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    parsed = pd.to_datetime(text, errors='coerce')
    if pd.isna(parsed):
        return pd.NA

    return parsed.strftime('%m/%d/%y')


def main():

    parser = ArgumentParser()
    parser.add_argument('-n', help="enable dry run",
                        action='store_true', default=False)
    parser.add_argument('-s', required=True,
                        help="Top folder from which to read results")
    parser.add_argument('-a', required=True, help="RedCap token")
    parser.add_argument('-r', required=True, help="report id to merge event with")
    args = parser.parse_args()
    dry_run = args.n
    parent_folder = args.s
    TOKEN = args.a
    report_id = args.r

    report = get_report(TOKEN, report_id)
    report = report.rename(columns={
        'mdt_date': 'visit_date'
    })
    report = report.drop(labels=['mri_date', 'np_date'], axis=1)
    report['visit_date'] = report['visit_date'].apply(normalize_report_date)
    report.to_csv('report.csv',index=False)
    # report['mdt_date'] = report['mdt_date'].apply()

    dataframe = Generate_Report(parent_folder)
    dataframe = dataframe.rename(columns={
        'id_participant': 'record_id'
    })

    dataframe = pd.merge(report, dataframe, 'inner', on=['visit_date', 'record_id'])
    dataframe = dataframe.sort_values(by=['record_id'])
    dataframe.to_csv('./output.csv', index=False)

    dataframe = dataframe.drop(labels=['id_session', 'visit_date'], axis=1)

    if not dry_run:
        data = dataframe.to_csv(index=False)
        upload_redcap(data,TOKEN)


if __name__ == "__main__":
    main()
