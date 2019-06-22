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
    con.cursor().execute("SET SCHEMA '{}'".format('Timestamps'))
except:
    pass