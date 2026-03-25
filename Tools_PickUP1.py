# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:23 2026

@author: Jesus Coss
"""

import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def GraficarOcupacion(df_plot):
    # 1. Definición del orden lógico (Pasado -> Presente)
    orden_rangos = [
        '+90 días', '2-3 meses', '1-2 meses', '2-4 semanas', '1-2 semanas', 
        '3-7 días', '1-3 días', '12-24 hrs', '6-12 hrs', '3-6 hrs', 
        '1-3 hrs', '0-1 hrs', 'Post-Salida'
    ]
    
    # Trabajamos sobre una copia para no afectar el dataframe original fuera de la función
    df_temp = df_plot.copy()

    # 2. CONVERTIR A CATEGORÍA ORDENADA
    # Esto le dice a Python que '1-2 semanas' SIEMPRE va antes que '1-3 días'
    df_temp['RANGO_ANTICIPACION'] = pd.Categorical(
        df_temp['RANGO_ANTICIPACION'], 
        categories=orden_rangos, 
        ordered=True
    )

    # 3. ORDENAR FÍSICAMENTE EL DATAFRAME
    # Ordenamos por ruta y luego por el orden lógico que definimos arriba
    df_temp = df_temp.sort_values(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])

    # 4. Configuración estética
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # 5. Dibujar las líneas
    # Ahora sort=True (por defecto) funcionará perfecto porque el DF ya está ordenado
    plot = sns.lineplot(
        data=df_temp, 
        x='RANGO_ANTICIPACION', 
        y='OCUPACION_ACUMULADA_%', 
        hue='ORIGEN_DESTINO', 
        marker='o',
        linewidth=2,
        markersize=7
    )
    
    # 6. Ajustes finales de formato
    plt.title('Curva de Llenado de Autobús (Pickup) por Ruta', fontsize=16, fontweight='bold')
    plt.xlabel('Ventana de Anticipación (De mayor a menor)', fontsize=12)
    plt.ylabel('% Ocupación Acumulada Promedio', fontsize=12)
    
    # Rotar etiquetas para que no se amontonen
    plt.xticks(rotation=45)
    
    # Límites lógicos y rejilla
    plt.ylim(0, 105) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Leyenda fuera del gráfico para que no estorbe
    plt.legend(title='Rutas (Origen-Destino)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def GraficarOcupacionPorHora(df_plot, hora_seleccionada):
    # 1. Definición del orden lógico
    orden_rangos = [
        '+90 días', '2-3 meses', '1-2 meses', '2-4 semanas', '1-2 semanas', 
        '3-7 días', '1-3 días', '12-24 hrs', '6-12 hrs', '3-6 hrs', '1-3 hrs', '0-1 hrs'
    ]
    
    # 2. Filtrado
    df_filtrado = df_plot[df_plot['HORA_ENTERA'] == hora_seleccionada].copy()
    
    if df_filtrado.empty:
        print(f"❌ ERROR: No se encontraron datos para la hora {hora_seleccionada}:00")
        return

    # --- LA CORRECCIÓN CLAVE AQUÍ ---
    # Convertimos la columna a Categorical con el orden lógico predefinido
    df_filtrado['RANGO_ANTICIPACION'] = pd.Categorical(
        df_filtrado['RANGO_ANTICIPACION'], 
        categories=orden_rangos, 
        ordered=True
    )
    
    # Ordenamos el dataframe físicamente por Ruta y luego por el Rango de Anticipación
    df_filtrado = df_filtrado.sort_values(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])
    # --------------------------------

    # 3. Configuración estética
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # 4. Creación del gráfico
    # Ahora sort=True funcionará correctamente porque definimos el orden categórico
    plot = sns.lineplot(
        data=df_filtrado, 
        x='RANGO_ANTICIPACION', 
        y='OCUPACION_ACUMULADA_%', 
        hue='ORIGEN_DESTINO', 
        marker='o',
        linewidth=2.5,
        markersize=8,
        sort=True  
    )
    
    plt.title(f'Curva de Llenado (Pickup) - Salidas: {hora_seleccionada}:00 hrs', fontsize=16, fontweight='bold')
    plt.xlabel('Ventana de Anticipación (Venta previa)', fontsize=12)
    plt.ylabel('% Ocupación Acumulada', fontsize=12)
    
    # Ajustar el eje X para que no muestre rangos vacíos si no quieres, 
    # o dejarlos todos para ver el hueco de ventas.
    plt.xticks(rotation=45)
    plt.ylim(0, 105) 
    
    plt.legend(title='Rutas (Origen-Destino)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

