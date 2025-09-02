# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

selected_files_df = ... # Compute a Pandas dataframe to write into selected_files


# Write recipe outputs
selected_files = dataiku.Dataset("selected_files")
selected_files.write_with_schema(selected_files_df)
