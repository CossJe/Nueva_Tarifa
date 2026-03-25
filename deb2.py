# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.dynamic_pricing_data_loader import cargar_y_preparar_datos
from Tools.ExtractData import GetDataFromETL
import seaborn as sns

Data= GetDataFromETL(True)
df=Data.Frame
df['FECHA_CORRIDA']=pd.to_datetime(df['FECHA_CORRIDA'])
df['HORA_SALIDA_CORRIDA']=pd.to_datetime(df['HORA_SALIDA_CORRIDA'])


