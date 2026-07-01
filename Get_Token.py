#!/usr/bin/env python3
#*****************************************************************#
###Title                           Get Token
###Purpose                          Send Study Short Name to
###                                 return corresponding API
###                                 token
###Creation Date                    01/06/2022
###Created by                       Derek V. Taylor
###Last Modified                    01/06/2022
#*****************************************************************#
import os
from os import path
from os.path import join
import json
import sys
import pandas as pd
import argparse


def token_fetch(sname):
    try:
        kpath = join('Keys','API_Keys.csv')
        df_keys = pd.read_csv(kpath)
        Trow = df_keys.loc[df_keys['Project'] == sname]
        rkey = Trow.iloc[0]['Key']
        return rkey
    except:
        return 'Failedd to return token' + str(sys.exc_info()[1])