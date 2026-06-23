import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Librerías para Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# Para estadísticas y análisis
from scipy.stats import skew, pearsonr

# Añadir estas importaciones
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

# Configuración global
COLOR_PALETTE = {
	"primary": "#3366CC",
	"secondary": "#6699CC",
	"tertiary": "#99CCFF",
	"success": "#66CC99",
	"warning": "#FFCC66",
	"danger": "#FF6666",
	"neutral": "#262730"
}

# Configuración de página
st.set_page_config(
	page_title="Dashboard de Ventas Avanzado",
	page_icon="📊",
	layout="wide"
)

# ---- FUNCIONES PRINCIPALES DEL DASHBOARD ----
@st.cache_data(ttl=3600, max_entries=10)
def cargar_datos(ruta="Ventas_y_Comportamiento_de_Clientes.csv"):
	"""
	Carga y preprocesa los datos con validación robusta.
	
	Args:
		ruta (str): Ruta al archivo CSV
		
	Returns:
		pd.DataFrame: DataFrame con los datos preprocesados
	"""
	try:
		df = pd.read_csv("Ventas.csv")
		
		# Preprocesamiento de fechas
		if "Fecha_de_compra" in df.columns:
			df["Fecha_de_compra"] = pd.to_datetime(df["Fecha_de_compra"], errors="coerce")
		
		# Extraer Ciudad/País de Ubicación_cliente
		if "Ubicación_cliente" in df.columns and df["Ubicación_cliente"].str.contains(",").any():
			df[['Ciudad', 'País']] = df["Ubicación_cliente"].str.split(',', n=1, expand=True)
			df[['Ciudad', 'País']] = df[['Ciudad', 'País']].apply(lambda x: x.str.strip())
		
		# Alertas de datos faltantes
		missing_data = df.isna().sum()
		if missing_data.any():
			missing_cols = missing_data[missing_data > 0].index.tolist()
			st.warning(f"⚠️ Valores faltantes detectados en: {', '.join(missing_cols)}")
		
		return df
	except FileNotFoundError:
		st.error("❌ Error: Archivo no encontrado. Verifique la ruta del archivo.")
		return pd.DataFrame()
	except pd.errors.EmptyDataError:
		st.error("❌ Error: El archivo está vacío.")
		return pd.DataFrame()
	except pd.errors.ParserError:
		st.error("❌ Error: Problema al parsear el archivo CSV. Verifique el formato.")
		return pd.DataFrame()
	except Exception as e:
		st.error(f"❌ Error inesperado: {str(e)}")
		return pd.DataFrame()

def aplicar_filtros(df, filtros):
	"""
	Aplica los filtros seleccionados al DataFrame.
	
	Args:
		df (pd.DataFrame): DataFrame original
		filtros (dict): Diccionario con los filtros a aplicar
		
	Returns:
		pd.DataFrame: DataFrame filtrado
	"""
	if df.empty: 
		return df
	
	df_filtrado = df.copy()
	
	# Filtros por atributos categóricos
	for col in ['Ciudad', 'País', 'Producto', 'Categoría_producto']:
		if col in filtros and filtros[col]:
			df_filtrado = df_filtrado[df_filtrado[col].isin(filtros[col])]
	
	# Filtros numéricos y de fecha
	if 'ventas_rango' in filtros and isinstance(filtros['ventas_rango'], tuple) and len(filtros['ventas_rango']) == 2:
		df_filtrado = df_filtrado[df_filtrado["Total_venta"].between(*filtros['ventas_rango'])]
	
	if 'fecha_rango' in filtros and isinstance(filtros['fecha_rango'], list) and len(filtros['fecha_rango']) == 2:
		fecha_inicio, fecha_fin = pd.to_datetime(filtros['fecha_rango'][0]), pd.to_datetime(filtros['fecha_rango'][1])
		df_filtrado = df_filtrado[df_filtrado["Fecha_de_compra"].between(fecha_inicio, fecha_fin + timedelta(days=1))]
	
	return df_filtrado

# ---- FUNCIONES DE ANÁLISIS DINÁMICO ----
def generar_interpretacion_histograma(df):
	"""
	Genera insights dinámicos para el histograma de ventas.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		str: Interpretación del histograma
	"""
	if df.empty or "Total_venta" not in df.columns:
		return "No hay datos suficientes para el análisis."
	
	skewness = skew(df["Total_venta"])
	q1 = df["Total_venta"].quantile(0.25)
	q3 = df["Total_venta"].quantile(0.75)
	avg = df["Total_venta"].mean()
	
	interpretacion = ""
	if skewness > 1:
		interpretacion += f"- **Distribución muy sesgada a la derecha** (skew = {skewness:.2f}): Concentración de ventas bajas. "
		interpretacion += f"El 75% de transacciones son < ${q3:,.2f}. "
		interpretacion += "**Acción:** Paquetes premium para aumentar ticket promedio."
	elif skewness < -1:
		interpretacion += f"- **Distribución sesgada a la izquierda** (skew = {skewness:.2f}): Predominan ventas altas. "
		interpretacion += f"El 25% de transacciones superan ${q1:,.2f}. "
		interpretacion += "**Oportunidad:** Optimizar inventario para artículos premium."
	else:
		interpretacion += "- Distribución balanceada. **Recomendación:** Mantener estrategias actuales."
	
	interpretacion += f"\n- **Ticket promedio actual:** ${avg:,.2f} (Rango: ${df['Total_venta'].min():,.2f} - ${df['Total_venta'].max():,.2f})"
	return interpretacion

def generar_interpretacion_tendencia(df):
	"""
	Genera insights para la tendencia de ventas.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		str: Interpretación de la tendencia
	"""
	if df.empty or "Fecha_de_compra" not in df.columns:
		return "No hay datos de fechas para análisis."
	
	df_agg = df.groupby(pd.Grouper(key="Fecha_de_compra", freq='D'))["Total_venta"].sum().reset_index()
	if len(df_agg) < 2:
		return "Datos insuficientes para detectar tendencias."
	
	ultimos_dias = df_agg["Total_venta"].tail(2)
	
	# Verificar que haya al menos dos días para comparar
	if len(ultimos_dias) < 2:
		return "Datos insuficientes para comparar tendencias diarias."
	
	if ultimos_dias.iloc[-2] == 0:
		cambio = 100.0 if ultimos_dias.iloc[-1] > 0 else 0.0
	else:
		cambio = ((ultimos_dias.iloc[-1] - ultimos_dias.iloc[-2]) / ultimos_dias.iloc[-2]) * 100
	
	varianza = df_agg["Total_venta"].var()
	estacionalidad = "Alta variabilidad diaria" if varianza > (df_agg["Total_venta"].mean()**2) else "Estabilidad relativa"
	
	interpretacion = f"- **Último día:** ${ultimos_dias.iloc[-1]:,.2f} ({cambio:.1f}% vs día anterior). "
	interpretacion += f"- **Estacionalidad:** {estacionalidad} (Varianza: {varianza:,.2f}). "
	
	if cambio > 15:
		interpretacion += "📈 **Acción:** Aumentar stock en días pico."
	elif cambio < -15:
		interpretacion += "📉 **Revisar** causas de bajas repentinas."
	else:
		interpretacion += "📊 **Análisis:** Tendencia estable en el periodo."
		
	return interpretacion

def generar_interpretacion_top_productos(df):
	"""
	Analiza la dependencia de productos líderes.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		str: Interpretación sobre los productos top
	"""
	if df.empty or "Producto" not in df.columns:
		return "Datos insuficientes para análisis de productos."
	
	total_ventas = df["Total_venta"].sum()
	if total_ventas == 0:
		return "No hay ventas registradas en el período seleccionado."
		
	top_productos = df.groupby("Producto")["Total_venta"].sum()
	
	if len(top_productos) == 0:
		return "No hay datos de productos para analizar."
		
	top_producto = top_productos.nlargest(1)
	porcentaje = (top_producto.iloc[0] / total_ventas) * 100
	
	interpretacion = f"- **Producto líder:** {top_producto.index[0]} ({porcentaje:.1f}% del total). "
	if porcentaje > 40:
		interpretacion += "🚨 Alta dependencia. **Acción crítica:** Diversificar portafolio."
	elif porcentaje > 20:
		interpretacion += "⚠️ Dependencia moderada. **Sugerencia:** Promocionar productos secundarios."
	else:
		interpretacion += "✅ Distribución saludable. **Mantener estrategias.**"
	return interpretacion

def generar_interpretacion_correlacion(df):
	"""
	Analiza relación entre satisfacción y ventas.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		str: Interpretación de la correlación
	"""
	if df.empty or ("Satisfacción_cliente" not in df.columns) or ("Total_venta" not in df.columns):
		return "Datos insuficientes para análisis de correlación."
	
	# Verificar que hay suficientes datos válidos para calcular la correlación
	valid_data = df.dropna(subset=["Satisfacción_cliente", "Total_venta"])
	if len(valid_data) < 3:
		return "Datos insuficientes para un análisis estadístico confiable."
	
	corr, p_value = pearsonr(valid_data["Satisfacción_cliente"], valid_data["Total_venta"])
	interpretacion = f"- **Correlación:** {corr:.2f} (Valor-p: {p_value:.3f}). "
	
	if p_value < 0.05:
		if corr > 0.5:
			interpretacion += "🔗 Relación fuerte positiva. **Invertir** en experiencia del cliente."
		elif corr < -0.5:
			interpretacion += "🔗 Relación fuerte negativa. **Investigar** causas de insatisfacción."
		else:
			interpretacion += "🔍 Relación significativa pero débil. **Priorizar** otros factores."
	else:
		interpretacion += "📊 Sin correlación estadística. **Enfoque alternativo:** Precio o ubicación."
	return interpretacion

# ---- COMPONENTES DE GRÁFICOS ----
def crear_histograma_ventas(df):
	"""
	Histograma interactivo con tooltips y anotaciones.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		plotly.graph_objects.Figure: Figura de Plotly
	"""
	if df.empty or "Total_venta" not in df.columns: 
		return None
		
	fig = px.histogram(
		df, x="Total_venta", nbins=20, 
		title="Distribución de Ventas",
		color_discrete_sequence=[COLOR_PALETTE["primary"]],
		template="plotly_white"
	)
	fig.update_traces(
		hovertemplate="<b>Rango:</b> $%{x:,.2f} <br><b>Transacciones:</b> %{y}",
		marker_line_width=1,
		marker_line_color="white"
	)
	
	# Agregar línea de promedio
	media = df["Total_venta"].mean()
	fig.add_vline(
		x=media, 
		line_dash="dash", 
		line_color=COLOR_PALETTE["warning"],
		annotation_text=f"Media: ${media:.2f}",
		annotation_position="top right"
	)
	
	fig.update_layout(
		xaxis_title="Valor de Venta ($)",
		yaxis_title="Número de Transacciones",
		plot_bgcolor=COLOR_PALETTE["neutral"],
		hoverlabel=dict(bgcolor="black", font_size=12)
	)
	
	return fig

def crear_tendencia_ventas(df):
	"""
	Gráfico de línea con detección de picos.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		plotly.graph_objects.Figure: Figura de Plotly
	"""
	if df.empty or "Fecha_de_compra" not in df.columns: 
		return None
		
	df_agg = df.groupby(pd.Grouper(key="Fecha_de_compra", freq='D'))["Total_venta"].sum().reset_index()
	
	if df_agg.empty:
		return None
		
	fig = px.line(
		df_agg, x="Fecha_de_compra", y="Total_venta", 
		title="Tendencia de Ventas Diarias",
		color_discrete_sequence=[COLOR_PALETTE["success"]],
		markers=True
	)
	
	# Detectar y marcar picos importantes
	if len(df_agg) > 1:
		max_venta = df_agg["Total_venta"].max()
		min_venta = df_agg["Total_venta"].min()
		
		# Marcar máximo
		fecha_pico = df_agg.loc[df_agg["Total_venta"] == max_venta, "Fecha_de_compra"].iloc[0]
		fig.add_annotation(
			x=fecha_pico, y=max_venta,
			text=f"🔥 Pico: ${max_venta:,.2f}",
			showarrow=True,
			arrowhead=2,
			arrowcolor=COLOR_PALETTE["danger"],
			bgcolor=COLOR_PALETTE["warning"],
			bordercolor=COLOR_PALETTE["danger"]
		)
		
		# Marcar mínimo
		fecha_min = df_agg.loc[df_agg["Total_venta"] == min_venta, "Fecha_de_compra"].iloc[0]
		fig.add_annotation(
			x=fecha_min, y=min_venta,
			text=f"📉 Mínimo: ${min_venta:,.2f}",
			showarrow=True,
			arrowhead=2,
			arrowcolor=COLOR_PALETTE["primary"],
			bgcolor="white",
			bordercolor=COLOR_PALETTE["primary"]
		)
	
	fig.update_layout(
		xaxis_title="Fecha",
		yaxis_title="Ventas Totales ($)",
		plot_bgcolor=COLOR_PALETTE["neutral"],
		hoverlabel=dict(bgcolor="black", font_size=12)
	)
	
	return fig

def crear_top_productos(df):
	"""
	Gráfico de barras con anotaciones.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
		
	Returns:
		plotly.graph_objects.Figure: Figura de Plotly
	"""
	if df.empty or "Producto" not in df.columns: 
		return None
		
	# Utilizar solo los 10 productos principales
	df_agg = df.groupby("Producto")["Total_venta"].sum().nlargest(10).reset_index()
	
	if df_agg.empty:
		return None
		
	fig = px.bar(
		df_agg, y="Producto", x="Total_venta", 
		title="Top 10 Productos por Ventas",
		color="Total_venta",
		color_continuous_scale=px.colors.sequential.Blues,
		text=df_agg["Total_venta"].apply(lambda x: f"${x:,.2f}")
	)
	
	fig.update_layout(
		yaxis={'categoryorder':'total ascending'},
		xaxis_title="Ventas Totales ($)",
		yaxis_title="Producto",
		plot_bgcolor=COLOR_PALETTE["neutral"],
		coloraxis_showscale=False
	)
	
	fig.update_traces(
		textposition="outside",
		hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.2f}<extra></extra>"
	)
	
	if not df_agg.empty:
		fig.add_annotation(
			x=df_agg["Total_venta"].iloc[-1] * 1.05, 
			y=df_agg["Producto"].iloc[-1],
			text="🚀 Líder de ventas",
			font=dict(color="green", size=12),
			showarrow=False,
			xanchor='left'
		)
	
	return fig

def crear_ventas_por_categoria(df):
	"""
	Gráficos combinados con leyendas interactivas.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
	"""
	if df.empty or "Categoría_producto" not in df.columns: 
		st.warning("No hay datos de categorías disponibles.")
		return
		
	df_agg = df.groupby("Categoría_producto")["Total_venta"].sum().reset_index()
	
	if df_agg.empty:
		st.warning("No hay datos agregados de categorías para mostrar.")
		return
		
	col1, col2 = st.columns(2)
	
	with col1:
		fig1 = px.bar(
			df_agg, x="Categoría_producto", y="Total_venta",
			title="Ventas por Categoría",
			color="Categoría_producto",
			color_discrete_sequence=px.colors.qualitative.Pastel,
			text=df_agg["Total_venta"].apply(lambda x: f"${x:,.2f}")
		)
		fig1.update_traces(
			textposition="outside",
			hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.2f}<extra></extra>"
		)
		fig1.update_layout(
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
			xaxis_title="Categoría de Producto",
			yaxis_title="Ventas Totales ($)",
			plot_bgcolor=COLOR_PALETTE["neutral"]
		)
		st.plotly_chart(fig1, use_container_width=True)
	
	with col2:
		fig2 = px.pie(
			df_agg, names="Categoría_producto", values="Total_venta",
			title="Distribución Porcentual de Ventas por Categoría",
			hole=0.4,
			color_discrete_sequence=px.colors.qualitative.Pastel
		)
		fig2.update_traces(
			textinfo="percent+label",
			hovertemplate="<b>%{label}</b><br>Ventas: $%{value:,.2f}<br>Porcentaje: %{percent}<extra></extra>"
		)
		fig2.update_layout(
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
		)
		st.plotly_chart(fig2, use_container_width=True)

# ---- SECCIÓN DE MACHINE LEARNING SUPERVISADO ---

def train_ml_model(df):
    """
    Entrena y muestra el modelo de regresión utilizando todas las características 
    disponibles y métodos avanzados de ML.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos completos
    """
    st.markdown("""
    **Modelo Utilizado:** XGBoostRegressor con Optimización Avanzada  
    **Pipeline de Preprocesamiento:**  
    - **Variables Numéricas:** StandardScaler  
    - **Variables Categóricas:** OneHotEncoder optimizado para múltiples ciudades y países
    **Feature Engineering:**  
    - Extracción completa de variables temporales (Día, Hora, DiaSemana, Mes, EsFinDeSemana)
    - Interacciones entre características clave
    - Selección automática de características importantes
    **Optimización de Hiperparámetros:** GridSearchCV con validación cruzada estratificada
    """)
    
    # Validar suficientes datos
    if df.empty or len(df) < 10:
        st.warning("⚠️ Datos insuficientes para entrenar un modelo confiable.")
        return
    
    try:
        # Crear copia y eliminar filas sin target
        df_ml = df.dropna(subset=["Total_venta"]).copy()
        
        # 2. MEJORA: Feature Engineering temporal avanzado
        if "Fecha_de_compra" in df_ml.columns:
            df_ml.loc[:, "Fecha_de_compra"] = pd.to_datetime(df_ml["Fecha_de_compra"], errors="coerce")
            df_ml.loc[:, "Día"] = df_ml["Fecha_de_compra"].dt.day
            df_ml.loc[:, "Hora"] = df_ml["Fecha_de_compra"].dt.hour
            df_ml.loc[:, "DiaSemana"] = df_ml["Fecha_de_compra"].dt.dayofweek  # 0=Lunes, 6=Domingo
            df_ml.loc[:, "Mes"] = df_ml["Fecha_de_compra"].dt.month
            df_ml.loc[:, "EsFinDeSemana"] = df_ml["Fecha_de_compra"].dt.dayofweek >= 5  # True para sábado/domingo
            df_ml = df_ml.drop("Fecha_de_compra", axis=1)
        
        # 3. MEJORA: Ampliar conjunto de características consideradas
        potential_features = [
            "Ciudad", "País", "Producto", "Categoría_producto", 
            "Día", "Hora", "DiaSemana", "Mes", "EsFinDeSemana",
            "Método_de_pago", "Dispositivo_de_compra", "Fuente_de_tráfico", 
            "Tiempo_de_navegación", "Descuento_aplicado", "Satisfacción_cliente",
            "Precio_unitario", "Cantidad"
        ]
        
        # Filtrar solo las características disponibles en el dataset
        features = [col for col in potential_features if col in df_ml.columns]
        
        if not features:
            st.warning("⚠️ No hay características disponibles para entrenar el modelo.")
            return
            
        X = df_ml[features]
        y = df_ml["Total_venta"]
        
        # Mostrar información del dataset de entrenamiento
        st.info(f"Entrenando modelo con {len(X)} registros y {len(features)} características, incluyendo datos de {X['Ciudad'].nunique()} ciudades y {X['País'].nunique()} países.")
        
        # División en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Identificar variables numéricas y categóricas
        num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_features = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
        
        # 1. MEJORA: Preprocesamiento optimizado para variables categóricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_features)
            ],
            remainder='passthrough'  # Incluir columnas no especificadas
        )
        
        # 7. MEJORA: Selección de características sofisticada
        feature_selector = SelectFromModel(XGBRegressor(n_estimators=100, random_state=42))
        
        # 4. MEJORA: Interacciones entre características
        poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        # 5. MEJORA: Reducción de dimensionalidad para manejar variables one-hot
        dim_reduction = PCA(n_components=0.95)  # Mantener el 95% de la varianza
        
        # Pipeline completo
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selector", feature_selector),
            ("poly_features", poly_features),
            ("dim_reduction", dim_reduction),
            ("regressor", XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])
        
        # Optimización de hiperparámetros
        param_grid = {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [3, 6, 9],
            "regressor__learning_rate": [0.01, 0.1, 0.2],
            "regressor__subsample": [0.8, 1.0],
            "regressor__colsample_bytree": [0.8, 1.0]
        }
        
        # 6. MEJORA: Validación cruzada estratificada
        y_bins = pd.qcut(y_train, q=5, labels=False, duplicates='drop')  # Dividir en 5 grupos por valor
        
        with st.spinner("Entrenando modelo avanzado... Esto puede tomar algunos minutos."):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, param_grid, 
                cv=cv.split(X_train, y_bins), 
                scoring="neg_mean_squared_error", 
                n_jobs=-1, 
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
        best_model = grid_search.best_estimator_
        
        # Evaluación del modelo
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R²", f"{r2:.2f}")
        
        # Mostrar mejores hiperparámetros
        st.subheader("Mejor configuración del modelo")
        best_params = grid_search.best_params_
        st.json(best_params)
        
        # Análisis de importancia de características mediante permutación
        with st.spinner("Calculando importancia de características... Este análisis es detallado y puede tomar tiempo."):
            result = permutation_importance(
                best_model, X_test, y_test, 
                n_repeats=10, 
                random_state=42, 
                scoring='neg_mean_squared_error'
            )
            
        # Obtener nombres de características en el pipeline transformado
        # Hacemos el mejor esfuerzo para mapear los índices a nombres de características
        try:
            # Obtener nombres de características después del preprocesamiento
            preprocessor = best_model.named_steps["preprocessor"]
            
            # Intentar obtener nombres de características
            feature_names = []
            
            # Para características numéricas
            if num_features:
                feature_names.extend(num_features)
            
            # Para características categóricas transformadas con OneHotEncoder
            if cat_features:
                for cat in cat_features:
                    # Si tenemos los valores únicos, los usamos para construir nombres de características
                    unique_vals = X[cat].unique()
                    for val in unique_vals[1:]:  # Saltamos el primero porque usamos drop='first'
                        feature_names.append(f"{cat}_{val}")
            
            # Si no podemos obtener los nombres exactos, usamos índices
            if len(feature_names) == 0 or len(feature_names) != result.importances_mean.shape[0]:
                feature_names = [f"Feature_{i}" for i in range(result.importances_mean.shape[0])]
                
        except Exception as e:
            # Si ocurre algún error, usar nombres genéricos
            feature_names = [f"Feature_{i}" for i in range(result.importances_mean.shape[0])]
            st.warning(f"No se pudieron identificar nombres exactos de características: {str(e)}")
        
        # 8. MEJORA: Visualización mejorada de importancia de características
        importance_df = pd.DataFrame({
            "Feature": feature_names[:len(result.importances_mean)],
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)
        
        # Filtrar características con importancia significativa
        top_features = importance_df[importance_df["Importance"] > 0]
        
        # Visualizar importancia de características
        st.subheader("Importancia de Características")
        fig = px.bar(
            top_features.head(30),  # Ampliado a 30 características
            x="Importance", 
            y="Feature",
            orientation='h',
            title="Top 30 Características por Importancia",
            color="Importance",
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Importancia",
            yaxis_title="Característica",
            plot_bgcolor=COLOR_PALETTE["neutral"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de dispersión: Valores Reales vs. Predicciones
        fig_scatter = px.scatter(
            x=y_test, 
            y=y_pred,
            labels={"x": "Valores Reales", "y": "Predicciones"},
            title="Valores Reales vs. Predicciones",
            trendline="ols",
            trendline_color_override=COLOR_PALETTE["danger"]
        )
        
        # Agregar línea de referencia perfecta
        fig_scatter.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()], 
                y=[y_test.min(), y_test.max()],
                mode="lines", 
                line=dict(color=COLOR_PALETTE["primary"], dash="dash"),
                name="Predicción Perfecta"
            )
        )
        
        fig_scatter.update_layout(
            plot_bgcolor=COLOR_PALETTE["neutral"],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis de residuos
        residuos = y_test - y_pred
        fig_residuos = px.scatter(
            x=y_pred, 
            y=residuos,
            labels={"x": "Valores Predichos", "y": "Residuos"},
            title="Análisis de Residuos",
            color=abs(residuos),
            color_continuous_scale=px.colors.sequential.Reds
        )
        
        fig_residuos.add_hline(y=0, line_dash="dash", line_color="black")
        fig_residuos.update_layout(plot_bgcolor=COLOR_PALETTE["neutral"])
        st.plotly_chart(fig_residuos, use_container_width=True)
        
        # Proporcionar insights sobre el modelo
        st.subheader("Interpretación del Modelo Mejorado")
        
        if r2 > 0.8:
            st.success("🎯 Modelo de alta precisión (R² > 0.8). Las predicciones son altamente confiables.")
        elif r2 > 0.6:
            st.info("✅ Modelo con precisión aceptable (R² > 0.6). Útil para la mayoría de casos de uso prácticos.")
        else:
            st.warning("⚠️ Modelo con precisión limitada (R² < 0.6). Considerar recolectar más datos o revisar variables.")
        
        # Análisis detallado de factores de impacto
        st.subheader("Factores Determinantes del Modelo")
        
        if top_features.empty:
            st.warning("No se encontraron características con importancia significativa.")
        else:
            # Agrupar características por tipo
            cat_impact = any(cat in ''.join(top_features['Feature'].astype(str)) for cat in ["Categoría", "Producto"])
            geo_impact = any(geo in ''.join(top_features['Feature'].astype(str)) for geo in ["Ciudad", "País"])
            time_impact = any(time in ''.join(top_features['Feature'].astype(str)) for time in ["Día", "Hora", "DiaSemana", "Mes", "EsFinDeSemana"])
            payment_impact = "Método_de_pago" in ''.join(top_features['Feature'].astype(str))
            discount_impact = "Descuento" in ''.join(top_features['Feature'].astype(str))
            device_impact = "Dispositivo" in ''.join(top_features['Feature'].astype(str))
            
            # Mostrar factores determinantes agrupados
            st.markdown(f"**Principal factor determinante:** `{top_features['Feature'].iloc[0]}`")
            
            impact_factors = []
            
            if cat_impact:
                impact_factors.append("🏷️ **Categoría y Producto:** Alta influencia en el valor de venta. Considerar estrategias de precios diferenciadas por tipo de producto.")
                
            if geo_impact:
                impact_factors.append("🌍 **Ubicación Geográfica:** Significativa variación por ubicación. Analizar oportunidades de expansión en ciudades/países de alto rendimiento.")
                
            if time_impact:
                impact_factors.append("⏰ **Patrones Temporales:** Variaciones importantes según día/hora. Optimizar inventario y promociones según estos patrones.")
                
            if payment_impact:
                impact_factors.append("💳 **Métodos de Pago:** Influyen en el comportamiento de compra. Considerar incentivos para métodos preferidos.")
                
            if discount_impact:
                impact_factors.append("🏷️ **Descuentos:** Impacto significativo en el valor final. Optimizar estrategia de descuentos.")
                
            if device_impact:
                impact_factors.append("📱 **Dispositivo de Compra:** Muestra preferencias de usuario. Optimizar experiencia para dispositivos clave.")
            
            # Mostrar factores de impacto
            for factor in impact_factors:
                st.markdown(factor)
            
            # Añadir recomendaciones específicas basadas en los hallazgos
            st.subheader("Recomendaciones Basadas en el Modelo")
            
            rec_count = 1
            recommendations = []
            
            if cat_impact and len(top_features) > 0:
                top_cat = [f for f in top_features['Feature'] if "Categoría" in f or "Producto" in f]
                if top_cat:
                    recommendations.append(f"{rec_count}. **Priorizar categorías clave:** Enfoque en {top_cat[0].split('_')[-1] if '_' in top_cat[0] else top_cat[0]} que muestra mayor impacto en ventas.")
                    rec_count += 1
            
            if geo_impact and len(top_features) > 0:
                top_geo = [f for f in top_features['Feature'] if "Ciudad" in f or "País" in f]
                if top_geo:
                    recommendations.append(f"{rec_count}. **Expansión geográfica:** Dirigir esfuerzos hacia {top_geo[0].split('_')[-1] if '_' in top_geo[0] else top_geo[0]} con mayores valores de venta.")
                    rec_count += 1
            
            if time_impact:
                recommendations.append(f"{rec_count}. **Optimización temporal:** Ajustar precios y promociones según patrones de día/hora identificados por el modelo.")
                rec_count += 1
            
            if len(recommendations) == 0:
                recommendations.append("1. **Análisis adicional:** El modelo sugiere revisar más datos para identificar patrones claros.")
            
            for recommendation in recommendations:
                st.markdown(recommendation)
        
    except Exception as e:
        st.error(f"Error durante el entrenamiento del modelo: {str(e)}")
        st.exception(e)  # Mostrar detalles completos del error para facilitar la depuración

# ---- FUNCIONES AVANZADAS DE RECOMENDACIÓN ----
def generar_recomendaciones_estrategicas(df):
	"""
	Genera recomendaciones estratégicas basadas en tendencias
	y patrones detectados en los datos.
	
	Args:
		df (pd.DataFrame): DataFrame con los datos
	"""
	st.header("Recomendaciones Estratégicas Basadas en Datos")
	
	if df.empty:
		st.warning("No hay datos suficientes para generar recomendaciones.")
		return
	
	try:
		# Calcular métricas clave
		n_ventas = len(df)
		total_ingresos = df["Total_venta"].sum()
		ticket_promedio = df["Total_venta"].mean()
		
		# Tendencia de ventas (últimos 7 días si hay datos suficientes)
		if "Fecha_de_compra" in df.columns:
			df_fecha = df.copy()
			df_fecha["Fecha_de_compra"] = pd.to_datetime(df_fecha["Fecha_de_compra"])
			df_reciente = df_fecha.sort_values("Fecha_de_compra", ascending=False)
			
			# Obtener tendencia reciente
			tendencia = pd.DataFrame()
			if len(df_reciente) > 1:
				tendencia = df_reciente.groupby(pd.Grouper(key="Fecha_de_compra", freq='D'))["Total_venta"].sum().reset_index()
				tendencia = tendencia.sort_values("Fecha_de_compra")
				
				if len(tendencia) > 1:
					primer_dia = tendencia["Total_venta"].iloc[0]
					ultimo_dia = tendencia["Total_venta"].iloc[-1]
					cambio_porcentual = ((ultimo_dia - primer_dia) / primer_dia) * 100 if primer_dia > 0 else 0
		
		# Crear tarjetas de recomendación
		col1, col2 = st.columns(2)
		
		with col1:
			st.subheader("Optimización de Inventario 📦")
			
			if "Producto" in df.columns:
				top_productos = df.groupby("Producto")["Total_venta"].sum().sort_values(ascending=False)
				bottom_productos = df.groupby("Producto")["Total_venta"].sum().sort_values()
				
				if not top_productos.empty and not bottom_productos.empty:
					st.markdown(f"""
					### Recomendaciones:
					
					1. **Aumentar stock** de productos líderes:
						- {top_productos.index[0]} (${top_productos.iloc[0]:,.2f} en ventas)
						{f"- {top_productos.index[1]} (${top_productos.iloc[1]:,.2f} en ventas)" if len(top_productos) > 1 else ""}
					
					2. **Reevaluar o promocionar** productos de bajo rendimiento:
						- {bottom_productos.index[0]} (${bottom_productos.iloc[0]:,.2f} en ventas)
						{f"- {bottom_productos.index[1]} (${bottom_productos.iloc[1]:,.2f} en ventas)" if len(bottom_productos) > 1 else ""}
					
					3. **Balance óptimo de inventario:**
						- Mantener un 60% de inventario en productos líderes
						- Reducir al 20% el inventario en productos de bajo rendimiento
						- Destinar 20% a pruebas de nuevos productos
					""")
			else:
				st.info("Datos de productos no disponibles.")
		
		with col2:
			st.subheader("Estrategia de Precios 💰")
			
			if "Satisfacción_cliente" in df.columns and len(df) > 5:
				# Analizar correlación precio-satisfacción
				corr = df[["Total_venta", "Satisfacción_cliente"]].corr().iloc[0,1]
				
				if np.isnan(corr):
					st.info("Datos insuficientes para análisis de correlación precio-satisfacción.")
				else:
					st.markdown(f"""
					### Recomendaciones:
					
					1. **Elasticidad de precios:** 
						- Correlación precio-satisfacción: {corr:.2f}
						- {"Incrementar precios gradualmente (baja sensibilidad)" if corr > 0 else "Mantener precios competitivos (alta sensibilidad)"}
					
					2. **Ticket promedio actual:** ${ticket_promedio:.2f}
						- {"Potencial de aumento del 5-10%" if corr > 0 else "Potencial de paquetes combinados"}
					
					3. **Estrategia recomendada:**
						- {"Valor Premium: Enfatizar calidad y exclusividad" if corr > 0 else "Valor Eficiente: Resaltar relación calidad-precio"}
					""")
			else:
				# Recomendaciones genéricas si no hay datos de satisfacción
				st.markdown(f"""
				### Recomendaciones:
				
				1. **Ticket promedio actual:** ${ticket_promedio:.2f}
					- Experimentar con incrementos del 5% en productos líderes
				
				2. **Bundles y paquetes:**
					- Crear ofertas combinadas para aumentar ticket promedio
				   - Objetivo: Incrementar ticket en 15% (a ${ticket_promedio * 1.15:.2f})
				
				3. **Promociones dinámicas:**
					- Descuentos estratégicos en días de menor afluencia
					- Precios premium en horas/días pico
				""")
		
		# Recomendaciones basadas en geografía
		if "País" in df.columns or "Ciudad" in df.columns:
			st.subheader("Expansión Geográfica 🌎")
			
			geo_metric = "País" if "País" in df.columns else "Ciudad"
			top_locations = df.groupby(geo_metric)["Total_venta"].sum().sort_values(ascending=False)
			
			if not top_locations.empty:
				st.markdown(f"""
				### Oportunidades de Expansión:
				
				1. **Mercados de alto rendimiento:**
					- {top_locations.index[0]} (${top_locations.iloc[0]:,.2f})
					{f"- {top_locations.index[1]} (${top_locations.iloc[1]:,.2f})" if len(top_locations) > 1 else ""}
				
				2. **Estrategia de penetración:**
					- Aumentar marketing en {", ".join(top_locations.index[:3].tolist()) if len(top_locations) >= 3 else top_locations.index[0]}
					- Analizar factores de éxito en estos mercados
				
				3. **Expansión controlada:**
					- Seleccionar mercados similares a los de alto rendimiento
					- Implementar estrategia de entrada gradual con métricas de control
				""")
				
				# Mapa de calor si hay suficientes datos
				if len(top_locations) > 5:
					st.subheader(f"Mapa de Calor por {geo_metric}")
					fig = px.bar(
						top_locations.reset_index().head(10), 
						x=geo_metric, 
						y="Total_venta",
						title=f"Top 10 {geo_metric}s por Ventas",
						color="Total_venta",
						color_continuous_scale=px.colors.sequential.Viridis
					)
					st.plotly_chart(fig, use_container_width=True)
		
		# Pronóstico básico
		if "Fecha_de_compra" in df.columns and len(tendencia) > 3:
			st.subheader("Pronóstico de Ventas 📈")
			
			# Calcular tendencias y estacionalidad simple
			tendencia_positiva = tendencia["Total_venta"].iloc[-1] > tendencia["Total_venta"].iloc[0]
			
			st.markdown(f"""
			### Proyección a 30 días:
			
			1. **Tendencia actual:** {"Positiva 📈" if tendencia_positiva else "Negativa 📉"} 
			({cambio_porcentual:.1f}% en el período analizado)
			
			2. **Ventas proyectadas:**
			   - Próximos 7 días: ${total_ingresos * (1 + cambio_porcentual/100):.2f}
			   - Próximos 30 días: ${total_ingresos * 4 * (1 + cambio_porcentual/100):.2f}
			
			3. **Acciones recomendadas:**
			- {"Aumentar inventario en anticipación al crecimiento" if tendencia_positiva else "Optimizar inventario para minimizar costos"}
			- {"Invertir en marketing para acelerar crecimiento" if tendencia_positiva else "Implementar promociones para revertir tendencia"}
			""")
		
	except Exception as e:
		st.error(f"Error al generar recomendaciones: {str(e)}")

# ---- APLICACIÓN PRINCIPAL ----
def main():
	"""Función principal que configura y ejecuta el dashboard."""
	
	st.title("🚀 Dashboard de Ventas y Análisis Avanzado")
	
	st.markdown("""
	Este dashboard proporciona análisis detallado e inteligencia de negocios sobre ventas, 
	comportamiento de clientes y rendimiento de productos. 
	Incluye visualizaciones interactivas, insights automáticos y un modelo predictivo.
	""")
	
	# Cargar datos
	df = cargar_datos()
	
	if df.empty:
		st.error("No se pudieron cargar los datos. Por favor verifique el archivo de entrada.")
		return
	
	# Panel de filtros
	st.sidebar.header("Filtros")
	
	# Filtros por fecha si está disponible
	fecha_filtro = None
	if "Fecha_de_compra" in df.columns:
		min_date = df["Fecha_de_compra"].min().date()
		max_date = df["Fecha_de_compra"].max().date()
		fecha_filtro = st.sidebar.date_input(
			"Rango de fechas",
			value=[min_date, max_date],
			min_value=min_date,
			max_value=max_date
		)
	
	# Filtros por categorías disponibles
	filtros = {}
	
	if fecha_filtro:
		filtros["fecha_rango"] = fecha_filtro
	
	for col in ["Ciudad", "País", "Categoría_producto"]:
		if col in df.columns:
			opciones = df[col].unique().tolist()
			seleccion = st.sidebar.multiselect(f"Seleccionar {col}", opciones)
			if seleccion:
				filtros[col] = seleccion
	
	# Filtro por rango de ventas si está disponible
	if "Total_venta" in df.columns:
		min_venta = float(df["Total_venta"].min())
		max_venta = float(df["Total_venta"].max())
		ventas_rango = st.sidebar.slider(
			"Rango de ventas ($)",
			min_venta, max_venta, 
			(min_venta, max_venta)
		)
		filtros["ventas_rango"] = ventas_rango
	
	# Aplicar filtros
	df_filtrado = aplicar_filtros(df, filtros)
	
	# Mostrar estadísticas básicas
	st.subheader("Resumen de Datos")
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.metric(
			"Total Ventas", 
			f"${df_filtrado['Total_venta'].sum():,.2f}" if "Total_venta" in df_filtrado.columns else "N/A"
		)
	
	with col2:
		st.metric(
			"# Transacciones", 
			f"{len(df_filtrado):,}"
		)
	
	with col3:
		st.metric(
			"Ticket Promedio", 
			f"${df_filtrado['Total_venta'].mean():,.2f}" if "Total_venta" in df_filtrado.columns else "N/A"
		)
	
	with col4:
		if "Satisfacción_cliente" in df_filtrado.columns:
			st.metric(
				"Satisfacción", 
				f"{df_filtrado['Satisfacción_cliente'].mean():.1f}/5"
			)
		else:
			st.metric("Satisfacción", "N/A")
	
	# Sección de gráficos principales
	st.subheader("Análisis Visual")
	
	tab1, tab2, tab3 = st.tabs(["Distribución de Ventas", "Tendencias Temporales", "Análisis por Producto"])
	
	with tab1:
		col1, col2 = st.columns([2, 1])
		
		with col1:
			fig_histograma = crear_histograma_ventas(df_filtrado)
			if fig_histograma:
				st.plotly_chart(fig_histograma, use_container_width=True)
			else:
				st.warning("No hay datos suficientes para crear el histograma.")
				
		with col2:
			st.subheader("Insights: Distribución de Ventas")
			st.markdown(generar_interpretacion_histograma(df_filtrado))
	
	with tab2:
		col1, col2 = st.columns([2, 1])
		
		with col1:
			fig_tendencia = crear_tendencia_ventas(df_filtrado)
			if fig_tendencia:
				st.plotly_chart(fig_tendencia, use_container_width=True)
			else:
				st.warning("No hay datos temporales suficientes para mostrar tendencias.")
				
		with col2:
			st.subheader("Insights: Tendencias Temporales")
			st.markdown(generar_interpretacion_tendencia(df_filtrado))
	
	with tab3:
		col1, col2 = st.columns([2, 1])
		
		with col1:
			fig_productos = crear_top_productos(df_filtrado)
			if fig_productos:
				st.plotly_chart(fig_productos, use_container_width=True)
			else:
				st.warning("No hay datos suficientes para mostrar análisis por producto.")
				
		with col2:
			st.subheader("Insights: Análisis de Productos")
			st.markdown(generar_interpretacion_top_productos(df_filtrado))
	
	# Gráficos por categoría
	st.subheader("Análisis por Categoría")
	crear_ventas_por_categoria(df_filtrado)
	
	# Sección de correlación
	st.subheader("Análisis de Correlación: Satisfacción vs. Ventas")
	if "Satisfacción_cliente" in df_filtrado.columns and "Total_venta" in df_filtrado.columns:
		fig = px.scatter(
			df_filtrado, 
			x="Satisfacción_cliente", 
			y="Total_venta", 
			title="Correlación entre Satisfacción y Valor de Venta",
			color="Satisfacción_cliente",
			color_continuous_scale=px.colors.sequential.Viridis,
			opacity=0.7,
			trendline="ols"
		)
		fig.update_layout(
			xaxis_title="Satisfacción del Cliente (1-5)",
			yaxis_title="Valor de Venta ($)",
			plot_bgcolor=COLOR_PALETTE["neutral"]
		)
		st.plotly_chart(fig, use_container_width=True)
		st.markdown(generar_interpretacion_correlacion(df_filtrado))
	else:
		st.warning("Datos de satisfacción no disponibles para análisis de correlación.")
	
	# Sección de machine learning
	st.header("Modelo de ML Supervisado Mejorado y Recomendaciones Estratégicas")
	st.markdown("""
	Este modelo utiliza todas las características disponibles para predecir el valor de venta y generar recomendaciones.
    Haga clic en el botón para entrenar el modelo y obtener recomendaciones estratégicas (puede tomar varios minutos).
    """)

	if st.button("📊 Entrenar Modelo y Generar Recomendaciones", type="primary"):
		with st.spinner("Entrenando modelo avanzado y generando recomendaciones... Este proceso puede tomar varios minutos."):
			train_ml_model(df)
			st.header("Recomendaciones Estratégicas Basadas en Modelo")
			generar_recomendaciones_estrategicas(df_filtrado)
	else:
		st.info("Haga clic en el botón para iniciar el entrenamiento del modelo y obtener recomendaciones estratégicas.")

	# Información sobre los datos y filtros
	with st.expander("Información sobre los datos"):
		st.markdown(f"""
		**Número total de registros:** {len(df):,}  
		**Registros filtrados:** {len(df_filtrado):,}  
		**Período analizado:** {df_filtrado['Fecha_de_compra'].min().date() if 'Fecha_de_compra' in df_filtrado.columns else 'N/A'} a {df_filtrado['Fecha_de_compra'].max().date() if 'Fecha_de_compra' in df_filtrado.columns else 'N/A'}  
		**Filtros aplicados:** {", ".join([f"{k}: {v}" for k, v in filtros.items()]) if filtros else "Ninguno"}
		""")
		
		# Mostrar los primeros 5 registros
		st.subheader("Vista previa de datos")
		st.dataframe(df_filtrado.head())

if __name__ == "__main__":
	main()