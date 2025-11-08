# src/model_training.py (v2.0 - Data Science Senior)

import logging
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Tuple, Dict, Any

from utils.db_manager import get_db_connection
from utils.logging_config import setup_logging

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración
setup_logging()
logger = logging.getLogger(__name__)

MODEL_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "match_predictor_v2.joblib")
ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "label_encoder_v2.joblib")
METADATA_SAVE_PATH = os.path.join(MODEL_DIR, "model_metadata_v2.joblib")

# Crear directorio si no existe
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================== CARGA DE DATOS ====================
def load_data_from_db() -> pd.DataFrame:
    """
    Carga partidos finalizados desde la BD con información enriquecida.
    """
    logger.info("=" * 70)
    logger.info("FASE 1: CARGA DE DATOS")
    logger.info("=" * 70)
    
    query = """
    SELECT
        p.id as partido_id,
        p.fecha_partido,
        p.competicion_id,
        p.equipo_local_id,
        p.equipo_visitante_id,
        p.goles_local,
        p.goles_visitante,
        p.resultado,
        e_local.nombre as nombre_local,
        e_vis.nombre as nombre_visitante
    FROM partidos p
    INNER JOIN equipos e_local ON p.equipo_local_id = e_local.id
    INNER JOIN equipos e_vis ON p.equipo_visitante_id = e_vis.id
    WHERE p.estado = 'FINISHED' 
        AND p.resultado IS NOT NULL
        AND p.goles_local IS NOT NULL
        AND p.goles_visitante IS NOT NULL
    ORDER BY p.fecha_partido ASC;
    """
    
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.error("No se encontraron partidos en la base de datos")
            return pd.DataFrame()
        
        # Convertir fecha a datetime
        df['fecha_partido'] = pd.to_datetime(df['fecha_partido'])
        
        logger.info(f"✓ {len(df)} partidos cargados")
        logger.info(f"  Rango de fechas: {df['fecha_partido'].min()} a {df['fecha_partido'].max()}")
        logger.info(f"  Competiciones: {df['competicion_id'].nunique()}")
        logger.info(f"  Equipos únicos: {df['equipo_local_id'].nunique() + df['equipo_visitante_id'].nunique()}")
        
        # Distribución de resultados
        result_dist = df['resultado'].value_counts()
        logger.info(f"  Distribución de resultados:")
        for resultado, count in result_dist.items():
            pct = (count / len(df)) * 100
            logger.info(f"    {resultado}: {count} ({pct:.1f}%)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


# ==================== FEATURE ENGINEERING ====================
def calculate_team_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcula la forma (puntos en últimos N partidos) de cada equipo.
    
    CRÍTICO: Evita data leakage usando solo partidos ANTERIORES.
    """
    logger.info(f"Calculando forma de equipos (ventana de {window} partidos)...")
    
    # Crear DataFrame con todos los partidos de cada equipo
    home_games = df[['partido_id', 'fecha_partido', 'equipo_local_id', 'resultado']].copy()
    home_games['es_local'] = True
    home_games['puntos'] = home_games['resultado'].map({'H': 3, 'D': 1, 'A': 0})
    home_games.rename(columns={'equipo_local_id': 'equipo_id'}, inplace=True)
    
    away_games = df[['partido_id', 'fecha_partido', 'equipo_visitante_id', 'resultado']].copy()
    away_games['es_local'] = False
    away_games['puntos'] = away_games['resultado'].map({'H': 0, 'D': 1, 'A': 3})
    away_games.rename(columns={'equipo_visitante_id': 'equipo_id'}, inplace=True)
    
    # Concatenar y ordenar por fecha
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games.sort_values(['equipo_id', 'fecha_partido'], inplace=True)
    
    # Calcular forma rodante (SIN incluir el partido actual)
    all_games['forma'] = all_games.groupby('equipo_id')['puntos'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
    )
    
    # Calcular partidos jugados
    all_games['partidos_previos'] = all_games.groupby('equipo_id').cumcount()
    
    # Separar local y visitante
    home_form = all_games[all_games['es_local']][['partido_id', 'forma', 'partidos_previos']].copy()
    home_form.columns = ['partido_id', 'forma_local', 'partidos_previos_local']
    
    away_form = all_games[~all_games['es_local']][['partido_id', 'forma', 'partidos_previos']].copy()
    away_form.columns = ['partido_id', 'forma_visitante', 'partidos_previos_visitante']
    
    # Merge con el DataFrame original
    df = df.merge(home_form, on='partido_id', how='left')
    df = df.merge(away_form, on='partido_id', how='left')
    
    logger.info(f"  ✓ Forma calculada para {len(df)} partidos")
    
    return df


def calculate_head_to_head(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Calcula estadísticas de enfrentamientos directos previos.
    """
    logger.info(f"Calculando historial H2H (últimos {window} enfrentamientos)...")
    
    h2h_stats = []
    
    for idx, row in df.iterrows():
        local_id = row['equipo_local_id']
        vis_id = row['equipo_visitante_id']
        fecha = row['fecha_partido']
        
        # Buscar partidos previos entre estos equipos
        prev_matches = df[
            (df['fecha_partido'] < fecha) &
            (
                ((df['equipo_local_id'] == local_id) & (df['equipo_visitante_id'] == vis_id)) |
                ((df['equipo_local_id'] == vis_id) & (df['equipo_visitante_id'] == local_id))
            )
        ].tail(window)
        
        if len(prev_matches) == 0:
            h2h_stats.append({
                'h2h_victorias_local': 0,
                'h2h_empates': 0,
                'h2h_victorias_visitante': 0,
                'h2h_partidos': 0
            })
            continue
        
        # Contar victorias desde la perspectiva del equipo local actual
        victorias_local = 0
        empates = 0
        victorias_vis = 0
        
        for _, match in prev_matches.iterrows():
            if match['equipo_local_id'] == local_id:
                # El local actual jugó de local en ese partido
                if match['resultado'] == 'H':
                    victorias_local += 1
                elif match['resultado'] == 'D':
                    empates += 1
                else:
                    victorias_vis += 1
            else:
                # El local actual jugó de visitante en ese partido
                if match['resultado'] == 'A':
                    victorias_local += 1
                elif match['resultado'] == 'D':
                    empates += 1
                else:
                    victorias_vis += 1
        
        h2h_stats.append({
            'h2h_victorias_local': victorias_local,
            'h2h_empates': empates,
            'h2h_victorias_visitante': victorias_vis,
            'h2h_partidos': len(prev_matches)
        })
    
    h2h_df = pd.DataFrame(h2h_stats)
    df = pd.concat([df.reset_index(drop=True), h2h_df], axis=1)
    
    logger.info(f"  ✓ H2H calculado para {len(df)} partidos")
    
    return df


def calculate_home_away_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Calcula estadísticas de rendimiento en casa/fuera.
    """
    logger.info(f"Calculando estadísticas local/visitante (ventana de {window})...")
    
    # Estadísticas de local
    home_stats = []
    for idx, row in df.iterrows():
        local_id = row['equipo_local_id']
        fecha = row['fecha_partido']
        
        # Partidos previos como local
        prev_home = df[
            (df['equipo_local_id'] == local_id) &
            (df['fecha_partido'] < fecha)
        ].tail(window)
        
        if len(prev_home) > 0:
            wins = (prev_home['resultado'] == 'H').sum()
            draws = (prev_home['resultado'] == 'D').sum()
            gf = prev_home['goles_local'].sum()
            gc = prev_home['goles_visitante'].sum()
        else:
            wins = draws = gf = gc = 0
        
        home_stats.append({
            'victorias_local_casa': wins,
            'empates_local_casa': draws,
            'goles_favor_local_casa': gf,
            'goles_contra_local_casa': gc,
            'partidos_local_casa': len(prev_home)
        })
    
    # Estadísticas de visitante
    away_stats = []
    for idx, row in df.iterrows():
        vis_id = row['equipo_visitante_id']
        fecha = row['fecha_partido']
        
        # Partidos previos como visitante
        prev_away = df[
            (df['equipo_visitante_id'] == vis_id) &
            (df['fecha_partido'] < fecha)
        ].tail(window)
        
        if len(prev_away) > 0:
            wins = (prev_away['resultado'] == 'A').sum()
            draws = (prev_away['resultado'] == 'D').sum()
            gf = prev_away['goles_visitante'].sum()
            gc = prev_away['goles_local'].sum()
        else:
            wins = draws = gf = gc = 0
        
        away_stats.append({
            'victorias_visitante_fuera': wins,
            'empates_visitante_fuera': draws,
            'goles_favor_visitante_fuera': gf,
            'goles_contra_visitante_fuera': gc,
            'partidos_visitante_fuera': len(prev_away)
        })
    
    home_df = pd.DataFrame(home_stats)
    away_df = pd.DataFrame(away_stats)
    
    df = pd.concat([df.reset_index(drop=True), home_df, away_df], axis=1)
    
    logger.info(f"  ✓ Estadísticas local/visitante calculadas")
    
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    if df.empty:
        logger.error("DataFrame vacío")
        return pd.DataFrame()
    
    original_count = len(df)
    
    # 1. Forma de equipos
    df = calculate_team_form(df, window=5)
    
    # 2. Head-to-head
    df = calculate_head_to_head(df, window=10)
    
    # 3. Estadísticas local/visitante
    df = calculate_home_away_stats(df, window=10)
    
    # 4. Features derivadas
    logger.info("Creando features derivadas...")
    
    # Diferencia de forma
    df['diferencia_forma'] = df['forma_local'] - df['forma_visitante']
    
    # Ratios H2H
    df['h2h_ratio_local'] = df['h2h_victorias_local'] / (df['h2h_partidos'] + 1)
    
    # Promedio goles
    df['avg_gf_local'] = df['goles_favor_local_casa'] / (df['partidos_local_casa'] + 1)
    df['avg_gc_local'] = df['goles_contra_local_casa'] / (df['partidos_local_casa'] + 1)
    df['avg_gf_visitante'] = df['goles_favor_visitante_fuera'] / (df['partidos_visitante_fuera'] + 1)
    df['avg_gc_visitante'] = df['goles_contra_visitante_fuera'] / (df['partidos_visitante_fuera'] + 1)
    
    # Experiencia (partidos jugados)
    df['experiencia_local'] = df['partidos_previos_local']
    df['experiencia_visitante'] = df['partidos_previos_visitante']
    
    # 5. Limpieza: eliminar primeros partidos sin suficiente historia
    min_partidos = 5
    df_clean = df[
        (df['partidos_previos_local'] >= min_partidos) &
        (df['partidos_previos_visitante'] >= min_partidos)
    ].copy()
    
    # Eliminar NaNs
    df_clean.dropna(inplace=True)
    
    removed = original_count - len(df_clean)
    logger.info(f"  ✓ {removed} partidos eliminados (datos insuficientes)")
    logger.info(f"  ✓ {len(df_clean)} muestras listas para entrenamiento")
    
    return df_clean


# ==================== ENTRENAMIENTO ====================
def select_features() -> list:
    """Define las features a usar en el modelo."""
    features = [
        # Forma
        'forma_local',
        'forma_visitante',
        'diferencia_forma',
        
        # H2H
        'h2h_victorias_local',
        'h2h_empates',
        'h2h_victorias_visitante',
        'h2h_ratio_local',
        
        # Local/Visitante
        'victorias_local_casa',
        'empates_local_casa',
        'victorias_visitante_fuera',
        'empates_visitante_fuera',
        
        # Goles
        'avg_gf_local',
        'avg_gc_local',
        'avg_gf_visitante',
        'avg_gc_visitante',
        
        # Experiencia
        'experiencia_local',
        'experiencia_visitante'
    ]
    
    return features


def train_model(df: pd.DataFrame) -> Tuple[Any, Any, Dict]:
    """
    Entrena y evalúa el modelo con validación cruzada.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 3: ENTRENAMIENTO Y EVALUACIÓN")
    logger.info("=" * 70)
    
    features = select_features()
    target = 'resultado'
    
    logger.info(f"Features seleccionadas: {len(features)}")
    for feat in features:
        logger.info(f"  • {feat}")
    
    X = df[features]
    y = df[target]
    
    # Codificar target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Clases: {list(label_encoder.classes_)}")
    
    # Split temporal (respetando orden cronológico)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y_encoded[:split_idx]
    y_test = y_encoded[split_idx:]
    
    logger.info(f"Split temporal:")
    logger.info(f"  Entrenamiento: {len(X_train)} muestras")
    logger.info(f"  Prueba: {len(X_test)} muestras")
    
    # Validación cruzada en train set
    logger.info("")
    logger.info("Validación cruzada (5-fold)...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Probar múltiples modelos
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model_name = None
    best_score = 0
    best_model = None
    
    for model_name, model in models.items():
        logger.info(f"  Evaluando {model_name}...")
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        
        logger.info(f"    Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = model_name
            best_model = model
    
    logger.info(f"  ✓ Mejor modelo: {best_model_name}")
    
    # Entrenar modelo final
    logger.info("")
    logger.info(f"Entrenando {best_model_name} en todo el train set...")
    best_model.fit(X_train, y_train)
    
    # Evaluar en test set
    logger.info("")
    logger.info("Evaluación en test set:")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"  Accuracy: {test_accuracy:.4f}")
    logger.info(f"  F1-Score (weighted): {test_f1:.4f}")
    
    # Reporte por clase
    logger.info("")
    logger.info("Reporte de clasificación:")
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        digits=4
    )
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    
    # Matriz de confusión
    logger.info("")
    logger.info("Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"  Predichas →    H     D     A")
    for i, clase in enumerate(label_encoder.classes_):
        logger.info(f"  {clase} (real)   {cm[i][0]:>4}  {cm[i][1]:>4}  {cm[i][2]:>4}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        logger.info("")
        logger.info("Importancia de features (Top 10):")
        importances = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(10).iterrows():
            logger.info(f"  {row['feature']:.<40} {row['importance']:.4f}")
    
    # Metadata
    metadata = {
        'model_type': best_model_name,
        'features': features,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'cv_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'train_date': datetime.now().isoformat(),
        'classes': list(label_encoder.classes_)
    }
    
    return best_model, label_encoder, metadata


def save_model(model: Any, encoder: Any, metadata: Dict):
    """Guarda modelo, encoder y metadata."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 4: GUARDADO DE MODELO")
    logger.info("=" * 70)
    
    joblib.dump(model, MODEL_SAVE_PATH)
    logger.info(f"✓ Modelo guardado: {MODEL_SAVE_PATH}")
    
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    logger.info(f"✓ Encoder guardado: {ENCODER_SAVE_PATH}")
    
    joblib.dump(metadata, METADATA_SAVE_PATH)
    logger.info(f"✓ Metadata guardada: {METADATA_SAVE_PATH}")


# ==================== MAIN ====================
if __name__ == '__main__':
    start_time = datetime.now()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " PIPELINE DE ENTRENAMIENTO DE MODELO v2.0 ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    # 1. Cargar datos
    df = load_data_from_db()
    
    if df.empty:
        logger.error("Pipeline abortado: no hay datos")
        exit(1)
    
    # 2. Feature engineering
    df_features = feature_engineering(df)
    
    if df_features.empty:
        logger.error("Pipeline abortado: feature engineering falló")
        exit(1)
    
    # 3. Entrenar modelo
    model, encoder, metadata = train_model(df_features)
    
    # 4. Guardar
    save_model(model, encoder, metadata)
    
    # Resumen final
    elapsed = datetime.now() - start_time
    
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " RESUMEN FINAL ".center(68) + "║")
    logger.info("╠" + "═" * 68 + "╣")
    logger.info(f"║  Tiempo total: {elapsed.total_seconds():.2f}s".ljust(69) + "║")
    logger.info(f"║  Modelo: {metadata['model_type']}".ljust(69) + "║")
    logger.info(f"║  Features: {len(metadata['features'])}".ljust(69) + "║")
    logger.info(f"║  Accuracy (CV): {metadata['cv_accuracy']:.4f}".ljust(69) + "║")
    logger.info(f"║  Accuracy (Test): {metadata['test_accuracy']:.4f}".ljust(69) + "║")
    logger.info(f"║  F1-Score (Test): {metadata['test_f1']:.4f}".ljust(69) + "║")
    logger.info("╠" + "═" * 68 + "╣")
    logger.info("║  Estado: ✓ COMPLETADO EXITOSAMENTE".ljust(69) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    exit(0)
