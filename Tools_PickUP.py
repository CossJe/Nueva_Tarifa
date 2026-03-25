# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:58:29 2026

@author: Jesus Coss
"""
import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def calcular_costos_operativos(distancia, costo_km, capacidad, ocupacion_esperada, margen_utilidad=0.16):
    """
    Fase 1: Cálculo del Costo Base y Precio Piso.
    Esta función calcula cuánto debe costar cada boleto para cubrir
    los gastos de la corrida y asegurar una utilidad mínima.
    """
    
    # Cálculo del costo total operativo (Gasto fijo)
    costo_total = distancia * costo_km
    
    # Número de asientos que se proyectan vender (Capacidad * Ocupación)
    asientos_vendidos = capacidad * ocupacion_esperada
    
    # Validación: Evitar división por cero si la ocupación es nula
    if asientos_vendidos <= 0:
        return {"error": "La ocupación esperada debe ser mayor a 0 para calcular un precio por asiento."}
    
    # Precio mínimo por asiento para cubrir el costo operativo + margen de utilidad
    # Fórmula: (Costo Total * (1 + Margen)) / Asientos Vendidos
    precio_piso = (costo_total * (1 + margen_utilidad)) / asientos_vendidos
    
    return {
        "costo_total": round(costo_total, 2),
        "precio_piso_minimo": round(precio_piso, 2),
        "asientos_a_vender": int(asientos_vendidos)
    }

def optimizar_mix_pasajeros_real(costo_operativo, capacidad_total, tarifa_adulto, descuentos):
    # --- LIMPIEZA DE DATOS (Evita el TypeError) ---
    def a_escalar(valor):
        # Si es una lista, array o Serie de pandas, extrae el primer elemento
        if hasattr(valor, '__iter__') and not isinstance(valor, dict):
            return float(valor[0])
        return float(valor)

    try:
        costo_operativo = a_escalar(costo_operativo)
        capacidad_total = int(a_escalar(capacidad_total))
        tarifa_adulto = a_escalar(tarifa_adulto)
    except Exception as e:
        return f"Error en el formato de los datos de entrada: {e}"

    # --- INICIO DEL MODELO ---
    prob = pulp.LpProblem("Llenar_Autobus_Rentable", pulp.LpMaximize)

    # Variables
    adultos = pulp.LpVariable('Adultos', lowBound=0, cat='Integer')
    
    # Variables de descuento con mínimos de ley
    vars_descuento = {}
    for tipo, info in descuentos.items():
        # Obligar a que INSEN y Niños sean >= 1
        piso_minimo = 1 if tipo.lower() in ['insen', 'ninos'] else 0
        
        vars_descuento[tipo] = pulp.LpVariable(
            tipo, 
            lowBound=piso_minimo, 
            upBound=info['limite'], 
            cat='Integer'
        )

    # --- FUNCIÓN OBJETIVO ---
    # Maximizamos pasajeros totales. 
    # Le damos un peso ínfimo extra al adulto (0.001) solo para priorizar ingreso en empates
    prob += adultos + pulp.lpSum([vars_descuento[t] for t in descuentos])

    # --- RESTRICCIONES ---
    # 1. Ingreso total >= costo operativo
    # Aquí es donde fallaba antes si tarifa_adulto era una lista
    ingreso_total = (adultos * tarifa_adulto) + \
                    pulp.lpSum([vars_descuento[t] * (tarifa_adulto * descuentos[t]['factor']) for t in descuentos])
    
    prob += ingreso_total >= costo_operativo
    
    # 2. Capacidad máxima
    prob += adultos + pulp.lpSum([vars_descuento[t] for t in descuentos]) <= capacidad_total

    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Verificación de resultados
    if pulp.LpStatus[prob.status] != 'Optimal':
        return "Infactible"
        
    return {
        "Adultos": int(adultos.varValue),
        "Descuentos": {t: int(vars_descuento[t].varValue) for t in descuentos},
        "Total_Pasajeros": int(pulp.value(adultos + pulp.lpSum([vars_descuento[t] for t in descuentos]))),
        "Ingreso_Final": round(pulp.value(ingreso_total), 2),
        "Costo_a_Cubrir": costo_operativo,
        "Estado": pulp.LpStatus[prob.status]
    }

def optimizar_precio(distancia, costo_km, capacidad, ocupacion_esperada):
    margen_utilidad=0.16
    # Cálculo del costo total operativo (Gasto fijo)
    costo_total = distancia * costo_km
    
    # Número de asientos que se proyectan vender (Capacidad * Ocupación)
    asientos_parciales = capacidad * ocupacion_esperada

    # Precio mínimo parcial por asiento para cubrir el costo operativo + margen de utilidad
    precio_parcial = (costo_total * (1 + margen_utilidad)) / asientos_parciales
    # Precio mínimo total por asiento para cubrir el costo operativo + margen de utilidad si vendes todos los asientos
    precio_total = (costo_total * (1 + margen_utilidad)) / capacidad
    
    limites_ley = {
        'estudiantes': {'factor': 0.85, 'limite': 2},
        'insen': {'factor': 0.5, 'limite': 2},
        'maestros': {'factor': 0.75, 'limite': 2},
        'ninos': {'factor': 0.5, 'limite': 2},
        'empleado_gobierno': {'factor': 0.90, 'limite': 2}
    }
    
    precios = np.round(np.linspace(precio_total, precio_parcial, 200),0)
    preciosf=[]
    for prize in precios:
        resultado = optimizar_mix_pasajeros_real(costo_total, asientos_parciales, prize, limites_ley)
        if not resultado == "Infactible":
            preciosf.append(prize)
    return min(preciosf)

def GetPrize(Tarifa,PrecioPiso, Rango):
    orden_rangos = [
        '+90 días', '2-3 meses', '1-2 meses', '2-4 semanas', '1-2 semanas', 
        '3-7 días', '1-3 días', '12-24 hrs', '6-12 hrs', '3-6 hrs', '1-3 hrs', '0-1 hrs'
    ]

    if Rango in orden_rangos[:4]:
        return float(Tarifa)
    elif Rango in orden_rangos[4:9]:
        return 0.45*float(Tarifa)+0.55*PrecioPiso
    elif Rango in orden_rangos[9:]:
        return PrecioPiso
    
def procesar_ocupacion_dinamica_v2(df_input, col_capacidad='Q'):
    # 1. Configuración de parámetros
    bins = [-np.inf, 0, 1, 3, 6, 12, 24, 72, 168, 336, 720, 1440, 2160, np.inf]
    labels = ['Post-Salida', '0-1 hrs', '1-3 hrs', '3-6 hrs', '6-12 hrs', '12-24 hrs', 
              '1-3 días', '3-7 días', '1-2 semanas', '2-4 semanas', '1-2 meses', '2-3 meses', '+90 días']
    orden_rangos = labels[::-1]

    # Pre-cálculo y optimización
    df_input = df_input.copy()
    df_input['Q'] = df_input['PAX_SUBEN'] + df_input['DISPONIBILIDAD_TRAMO']
    df_input.loc[df_input['Q'] <= 0, 'Q'] = df_input['CAPACIDAD_ASIENTOS_TRAMO']
    df_input['HORA_ENTERA'] = df_input['HORA_ENTERA'].astype(int).astype('int8')

    # 2. Pipeline de procesamiento
    df_result = (
        df_input.drop_duplicates().assign(
            CV_CORRIDA = lambda x: x['CV_CORRIDA'].astype(str),
            RANGO_ANTICIPACION = lambda x: pd.Categorical(
                pd.cut(x['HORAS_ANTICIPACION'], bins=bins, labels=labels),
                categories=orden_rangos, ordered=True
            )
        )
        .query("BOLETOS_VEND > 0 & RANGO_ANTICIPACION != 'Post-Salida'")
        .groupby(['ORIGEN_DESTINO', 'CV_CORRIDA', 'FECHA_CORRIDA', 'HORA_ENTERA', 'RANGO_ANTICIPACION'], observed=True)
        .agg(
            BOLETOS_VEND=('BOLETOS_VEND', 'sum'), 
            CAPACIDAD_ORIGINAL=(col_capacidad, 'mean'),
            Ingreso=('VENTA','sum'),
            Tarifa=('TARIFA_BASE_TRAMO','sum')
        )
        .reset_index()
        .sort_values(['CV_CORRIDA', 'FECHA_CORRIDA', 'HORA_ENTERA', 'RANGO_ANTICIPACION'])
    )

    
    # 3. Cálculos Dinámicos - SUMA ACUMULADA DE OCUPACIÓN
    group = df_result.groupby(['CV_CORRIDA', 'FECHA_CORRIDA', 'HORA_ENTERA'])

    # Venta acumulada (Numerador)
    df_result['VENTA_ACUMULADA'] = group['BOLETOS_VEND'].cumsum()

    # Venta previa para capacidad dinámica (Informativo)
    df_result['VENTA_PREVIA_ACUM'] = group['BOLETOS_VEND'].shift(1, fill_value=0).cumsum()
    df_result['CAPACIDAD_DINAMICA'] = (df_result['CAPACIDAD_ORIGINAL'] - df_result['VENTA_PREVIA_ACUM']).clip(lower=0)

    # --- CORRECCIÓN: OCUPACIÓN ACUMULADA SIGUIENDO EL CRITERIO DE VENTA ---
    # 1. Calculamos cuánto representa ESTA venta del rango actual sobre el TOTAL del bus
    df_result['OCUPACION_INDIVIDUAL_%'] = (df_result['BOLETOS_VEND'] / df_result['CAPACIDAD_ORIGINAL']) * 100
    
    # 2. Hacemos la suma acumulada de esos porcentajes por grupo
    df_result['OCUPACION_ACUMULADA_%'] = group['OCUPACION_INDIVIDUAL_%'].cumsum()
    # ----------------------------------------------------------------------

    # 4. Resumen para gráfica
    df_plot = (
        df_result.groupby(['ORIGEN_DESTINO', 'HORA_ENTERA', 'RANGO_ANTICIPACION'], observed=True)
        ['OCUPACION_ACUMULADA_%']
        .mean()
        .reset_index()
    )

    df_plot1 = (
        df_result.groupby(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'], observed=True)
        ['OCUPACION_ACUMULADA_%']
        .mean()
        .reset_index()
    )

    
    return df_result, df_plot,df_plot1

def GetData4Hour(df_plot, hora_seleccionada):
    df_filtrado = df_plot[df_plot['HORA_ENTERA'] == hora_seleccionada].copy()
    df_temp = df_filtrado.copy()
    orden_rangos = [
        '+90 días', '2-3 meses', '1-2 meses', '2-4 semanas', '1-2 semanas', 
        '3-7 días', '1-3 días', '12-24 hrs', '6-12 hrs', '3-6 hrs', '1-3 hrs', '0-1 hrs'
    ]
    df_temp['RANGO_ANTICIPACION'] = pd.Categorical(
            df_temp['RANGO_ANTICIPACION'], 
            categories=orden_rangos, 
            ordered=True
        )
    return df_temp.sort_values(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])

def crear_indice_maestro(df, columnas_llave):
    """
    df: Puede ser df_plot (con hora) o df_plot1 (sin hora).
    columnas_llave: Lista de columnas que definen la curva (ej. ['ORIGEN_DESTINO', 'HORA_ENTERA', 'RANGO_ANTICIPACION'])
    """
    # Creamos un diccionario donde la llave es una tupla con los valores de las columnas
    return df.set_index(columnas_llave)['OCUPACION_ACUMULADA_%'].to_dict()

def proyectar_velocidad_general(datos_corrida_viva, indice_maestro, columnas_llave):
    """
    datos_corrida_viva: Un dict o fila con la info actual (Ruta, Hora, Ocupacion, Rango).
    indice_maestro: El diccionario creado arriba.
    columnas_llave: Las mismas columnas que usaste para el índice.
    """
    # 1. Construir la llave de búsqueda dinámicamente
    llave_busqueda = tuple(datos_corrida_viva[col] for col in columnas_llave)
    
    # 2. Obtener el ADN histórico
    esperado_historico = indice_maestro.get(llave_busqueda)
    
    if esperado_historico is None or esperado_historico == 0:
        return None # O manejar como "Sin Datos"

    # 3. Cálculo de Velocidad (Pace)
    ocupacion_hoy = datos_corrida_viva['OCUPACION_ACUMULADA_%']
    pace_index = ocupacion_hoy / esperado_historico
    
    # 4. Proyección de cierre
    cierre_estimado = min(100.0, 100.0 * pace_index)
    
    return {
        "Llave": llave_busqueda,
        "Pace": round(pace_index, 2),
        "Proyeccion": round(cierre_estimado, 2),
        "Desviacion": round(pace_index - 1, 2) # Negativo es "Atrasado", Positivo "Adelantado"
    }

def agente_revenue_management(corrida_info, velocidad):
    # 1. Tomar los datos calculados
    costo = corrida_info['costo_total']
    capacidad = corrida_info['asientos_a_vender']
    tarifa_adulto = corrida_info['PrecioOptimo']
    Pace= velocidad['Pace']
    
    # 2. Lógica de Ajuste: Definir límites de descuento según el Pace
    # Si el Pace es 0.8 (frío), permitimos más descuentos para llenar.
    # Si el Pace es 1.2 (caliente), restringimos descuentos al mínimo.
    
    if Pace < 0.9:
        print("¡Alerta: Corrida fría! Abriendo cupos de descuento al máximo.")
        # Aumentamos límites
        limites_activos = {
        'estudiantes': {'factor': 0.85, 'limite': 4},
        'insen': {'factor': 0.5, 'limite': 4},
        'maestros': {'factor': 0.75, 'limite': 4},
        'ninos': {'factor': 0.5, 'limite': 4},
        'empleado_gobierno': {'factor': 0.90, 'limite': 4}}
    elif Pace > 1.1:
        print("¡Corrida caliente! Limitando descuentos para maximizar tarifa plena.")
        #Sin descuentos
        limites_activos = {
        'estudiantes': {'factor': 0.85, 'limite': 0},
        'insen': {'factor': 0.5, 'limite': 1},
        'maestros': {'factor': 0.75, 'limite': 0},
        'ninos': {'factor': 0.5, 'limite': 0},
        'empleado_gobierno': {'factor': 0.90, 'limite': 0}}
    else:
        print("¡Corrida normal! ")
        # Normal
        limites_activos = {
        'estudiantes': {'factor': 0.85, 'limite': 2},
        'insen': {'factor': 0.5, 'limite': 2},
        'maestros': {'factor': 0.75, 'limite': 2},
        'ninos': {'factor': 0.5, 'limite': 2},
        'empleado_gobierno': {'factor': 0.90, 'limite': 2}}

    # 3. Fase 2: Ejecutar el Optimizador con los nuevos límites
    resultado_mix = optimizar_mix_pasajeros_real(costo, capacidad, tarifa_adulto, limites_activos)
    
    return {
        "Pace_Index": Pace,
        "Mix_Recomendado": resultado_mix}

def GetElasticity(df_detalle):
    # Calculamos el Precio Promedio Real por boleto en cada registro
    df_detalle['PRECIO_PROMEDIO_PAGADO'] = df_detalle['Ingreso'] / df_detalle['BOLETOS_VEND']
    # Agrupamos para obtener la Tarifa Promedio y Ocupación por Ruta y Rango
    # Agregamos 'FECHA_CORRIDA' (por mes o semana) para tener puntos de comparación
    df_detalle['MES'] = pd.to_datetime(df_detalle['FECHA_CORRIDA']).dt.to_period('M')
    
    # Ahora, cuando agrupes para la elasticidad, usa esta nueva variable
    df_elasticidad = df_detalle.groupby(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION', 'MES'], observed=True).agg({
        'PRECIO_PROMEDIO_PAGADO': 'mean', # Este es tu P real
        'OCUPACION_INDIVIDUAL_%': 'mean'  # Este es tu Q real
    }).reset_index()
    
    # Aseguramos el orden cronológico para que el pct_change tenga sentido
    df_elasticidad = df_elasticidad.sort_values(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION', 'MES'])
    
    # Calculamos la variación porcentual de P y Q
    # Agrupamos por Ruta y Rango para que no compare la última fila de una ruta con la primera de otra
    df_elasticidad['delta_P'] = df_elasticidad.groupby(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])['PRECIO_PROMEDIO_PAGADO'].pct_change()
    df_elasticidad['delta_Q'] = df_elasticidad.groupby(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])['OCUPACION_INDIVIDUAL_%'].pct_change()
    
    # Calcular la Elasticidad Punto a Punto
    df_elasticidad['elasticidad_punto'] = df_elasticidad['delta_Q'] / df_elasticidad['delta_P']
    
    #  Eliminar infinitos y nulos
    df_clean = df_elasticidad.replace([np.inf, -np.inf], np.nan).dropna(subset=['elasticidad_punto'])
    
    # Filtrar Outliers: En transporte, una elasticidad lógica suele estar entre -5 y 0.
    # Si te da +10, es ruido (el precio subió y la demanda también por un festival, por ejemplo).
    df_final_elasticidad = df_clean[df_clean['elasticidad_punto'].between(-5, 0)]
    
    # Usamos la mediana para ser más robustos contra valores extremos
    tabla_maestra_elasticidad = df_final_elasticidad.groupby(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])['elasticidad_punto'].median().reset_index()
    
    # Re-ordenar según tus etiquetas de anticipación
    orden_rangos = ['+90 días', '2-3 meses', '1-2 meses', '2-4 semanas', '1-2 semanas', 
                    '3-7 días', '1-3 días', '12-24 hrs', '6-12 hrs', '3-6 hrs', '1-3 hrs', '0-1 hrs']
    tabla_maestra_elasticidad['RANGO_ANTICIPACION'] = pd.Categorical(tabla_maestra_elasticidad['RANGO_ANTICIPACION'], categories=orden_rangos, ordered=True)
    tabla_maestra_elasticidad = tabla_maestra_elasticidad.sort_values(['ORIGEN_DESTINO', 'RANGO_ANTICIPACION'])
    
    tabla_maestra_elasticidad= tabla_maestra_elasticidad.dropna(subset=['RANGO_ANTICIPACION'])
    tabla_maestra_elasticidad['elasticidad_punto']= tabla_maestra_elasticidad['elasticidad_punto'].fillna(0)

    return tabla_maestra_elasticidad