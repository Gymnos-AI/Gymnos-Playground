import sys
import os
import pandas as pd
import subprocess
import argparse
import pdb
import pickle
from setup import setup_environment

# Make PostgreSQL Connection
engine = setup_environment.get_database()
try:
    con = engine.raw_connection()
    cursor = con.cursor()
    cursor.execute("SET SCHEMA '{}'".format('Timestamps'))
    print("attempting insert")
    query = "INSERT INTO Timestamps VALUES '06-21-2018', '06-22-2019'"
    query1 = "SELECT * FROM Timestamps"
    print("insert complete")
    cursor.execute(query)
    result = cursor.execute(query1)
    print(result)
    #result.close()
except:
    pass