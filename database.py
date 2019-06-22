import sys
import os
import pandas as pd
import subprocess
import argparse
import pdb
import pickle
from setup import setup_environment

def createTable():
    try:
        creation_query = "CREATE TABLE timestamps " \
                         "(inframe message_text " \
                         "outframe message_text " \
                         ")"

        # create the table
        print("attempting query 1")
        cursor.execute(creation_query)
        con.commit()
    except:
        print("Failed creation")
        pass

def insertTimestamp():
    try:
        query = "INSERT INTO gymnos (inframe, outframe) VALUES ('06-21-2018', '06-22-2019')"

        print("attempting insert")
        cursor.execute(query)
    except:
        print("Failed insertion")
        pass

def getTimestamp():
    try:
        query = "SELECT inframe, outframe FROM gymnos"

        print("attempting get")
        cursor.execute(query)

        # get results
        result = cursor.fetchall()

        #print("Finished")
        print("result is" + result)

        for i in result:
            print("1")
            print(result[0])

    except:
        print("Failed get")
        pass

# Make PostgreSQL Connection
engine = setup_environment.get_database()

try:
    con = engine.raw_connection()
    cursor = con.cursor()

    createTable()
    insertTimestamp()
    getTimestamp()

    #close the cursor
    cursor.close()

    #close the connection
    con.close()
except:
    print("failed")
    pass