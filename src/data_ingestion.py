# src/data_ingestion.py (v3.2 - Arquitecto Senior - Antibalas)

import logging
import time
import yaml
from datetime import datetime
from psycopg2 import extras, DatabaseError, IntegrityError
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps

from utils.db_manager import get_db_connection
from utils.api_client import get_data
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ==================== CONFIGURACIÓN ====================
def load_config(path: str = 'config/pipeline_config.yaml') -> Dict[str, Any]:
    """Carga configuración desde archivo YAML."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Archivo de configuración no encontrado: {path}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear YAML: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error inesperado cargando configuración: {e}", exc_info=True)
        exit(1)


def retry_on_failure(max_retries: int, delay: float):
    """
    Decorador para reintentar funciones que fallan.
    
    Args:
        max_retries: Número máximo de intentos
        delay: Segundos de espera entre intentos
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Intento {attempt}/{max_retries} falló para {func.__name__}: {e}. "
                            f"Reintentando en {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Función {func.__name__} falló después de {max_retries} intentos: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


# ==================== FUNCIONES DE VALIDACIÓN ====================
def map_resultado(winner: Optional[str]) -> Optional[str]:
    """
    Convierte el formato de winner de la API al ENUM de la BD.
    
    Args:
        winner: 'HOME_TEAM', 'AWAY_TEAM', 'DRAW', o None
    
    Returns:
        'H', 'A', 'D', o None
    """
    if not winner:
        return None
    
    mapping = {
        'HOME_TEAM': 'H',
        'AWAY_TEAM': 'A',
        'DRAW': 'D'
    }
    
    result = mapping.get(winner)
    if not result:
        logger.warning(f"Valor 'winner' desconocido: {winner}")
    
    return result


def validate_match_data(match: Dict, competition_code: str) -> Optional[Tuple]:
    """
    Valida y extrae datos de un partido.
    
    Returns:
        Tupla con datos del partido o None si inválido
    """
    # Validar campos requeridos
    required = ['id', 'utcDate', 'homeTeam', 'awayTeam', 'score']
    if not all(field in match for field in required):
        logger.debug(f"Partido {match.get('id', 'N/A')} - faltan campos")
        return None
    
    match_id = match['id']
    home_id = match['homeTeam'].get('id')
    away_id = match['awayTeam'].get('id')
    
    # Validar IDs de equipos
    if not home_id or not away_id:
        logger.debug(f"Partido {match_id} - IDs inválidos")
        return None
    
    if home_id == away_id:
        logger.warning(f"Partido {match_id} - equipo local = visitante (dato corrupto)")
        return None
    
    # Extraer goles
    score = match.get('score', {})
    full_time = score.get('fullTime', {})
    goles_local = full_time.get('home')
    goles_visitante = full_time.get('away')
    
    # Estado y resultado
    estado = match.get('status', 'SCHEDULED')
    resultado = map_resultado(score.get('winner'))
    
    # Validación estricta para partidos finalizados
    if estado == 'FINISHED':
        if goles_local is None or goles_visitante is None:
            logger.warning(f"Partido {match_id} - finalizado sin goles registrados")
            return None
        if not resultado:
            logger.warning(f"Partido {match_id} - finalizado sin resultado")
            return None
    
    # Parsear fecha
    try:
        fecha_partido = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
    except (ValueError, AttributeError, TypeError) as e:
        logger.debug(f"Partido {match_id} - fecha inválida: {e}")
        return None
    
    return (
        match_id,
        fecha_partido,
        competition_code,
        home_id,
        away_id,
        goles_local,
        goles_visitante,
        resultado,
        estado
    )


def validate_standing_data(row: Dict, competition_code: str) -> Optional[Tuple]:
    """
    Valida y extrae datos de una posición.
    
    Returns:
        Tupla con datos de posición o None si inválido
    """
    team_info = row.get('team')
    if not team_info or not team_info.get('id'):
        logger.debug("Fila de posición sin equipo válido")
        return None
    
    equipo_id = team_info['id']
    
    # Validar campos numéricos obligatorios
    required_nums = [
        'position', 'points', 'playedGames', 'won', 'draw', 'lost',
        'goalsFor', 'goalsAgainst', 'goalDifference'
    ]
    
    for field in required_nums:
        if row.get(field) is None:
            logger.debug(f"Equipo {equipo_id} - falta campo {field}")
            return None
    
    # Validar consistencia matemática
    jugados = row['playedGames']
    ganados = row['won']
    empatados = row['draw']
    perdidos = row['lost']
    
    if jugados != ganados + empatados + perdidos:
        logger.warning(
            f"Equipo {equipo_id} - inconsistencia: "
            f"jugados({jugados}) != ganados({ganados}) + empatados({empatados}) + perdidos({perdidos})"
        )
        return None
    
    # Validar diferencia de goles
    gf = row['goalsFor']
    gc = row['goalsAgainst']
    diff = row['goalDifference']
    
    if gf - gc != diff:
        logger.warning(
            f"Equipo {equipo_id} - diferencia de goles incorrecta: "
            f"{gf} - {gc} != {diff}"
        )
        return None
    
    return (
        competition_code,
        equipo_id,
        row['position'],
        row['points'],
        jugados,
        ganados,
        empatados,
        perdidos,
        gf,
        gc,
        diff
    )


# ==================== INGESTIÓN DE COMPETICIONES ====================
def ingest_competitions(config: Dict[str, Any]) -> bool:
    """
    Ingesta competiciones desde la API.
    
    Returns:
        True si exitosa, False en caso contrario
    """
    logger.info("=" * 70)
    logger.info("FASE 1: INGESTIÓN DE COMPETICIONES")
    logger.info("=" * 70)
    
    @retry_on_failure(config['api']['max_retries'], config['api']['retry_delay'])
    def fetch_competitions():
        return get_data("competitions")
    
    try:
        api_data = fetch_competitions()
    except Exception as e:
        logger.error(f"Error al obtener competiciones: {e}")
        return False
    
    if not api_data or 'competitions' not in api_data:
        logger.error("No se obtuvieron competiciones de la API")
        return False
    
    valid_competitions = []
    invalid_count = 0
    
    for comp in api_data['competitions']:
        code = comp.get('code')
        name = comp.get('name')
        country = comp.get('area', {}).get('name', 'N/A')
        
        if code and name:
            valid_competitions.append((code, name, country))
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"{invalid_count} competiciones inválidas descartadas")
    
    if not valid_competitions:
        logger.error("No hay competiciones válidas para insertar")
        return False
    
    logger.info(f"Competiciones válidas: {len(valid_competitions)}")
    
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO competiciones (id, nombre, pais)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                nombre = EXCLUDED.nombre,
                pais = EXCLUDED.pais;
        """
        
        extras.execute_values(
            cursor,
            sql,
            valid_competitions,
            page_size=config['database']['batch_size']
        )
        
        affected = cursor.rowcount
        conn.commit()
        
        logger.info(f"✓ {affected} competiciones procesadas")
        return True
        
    except DatabaseError as e:
        logger.error(f"Error de base de datos: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
        
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== INGESTIÓN DE EQUIPOS ====================
def ingest_teams_and_relations(config: Dict[str, Any], competition_codes: List[str]) -> Dict[str, Any]:
    """
    Ingesta equipos y sus relaciones con competiciones.
    
    Returns:
        Diccionario con estadísticas
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 2: INGESTIÓN DE EQUIPOS Y RELACIONES")
    logger.info("=" * 70)
    
    all_teams_data: Dict[int, Dict] = {}
    team_competition_relations = []
    failed_competitions = []
    stats_by_competition = {}
    
    @retry_on_failure(config['api']['max_retries'], config['api']['retry_delay'])
    def fetch_teams(code: str):
        return get_data(f"competitions/{code}/teams")
    
    for idx, code in enumerate(competition_codes, 1):
        logger.info(f"[{idx}/{len(competition_codes)}] Procesando: {code}")
        
        try:
            api_data = fetch_teams(code)
        except Exception as e:
            logger.error(f"  └─ ✗ Error: {e}")
            failed_competitions.append(code)
            continue
        
        if not api_data or 'teams' not in api_data:
            logger.warning(f"  └─ ✗ Sin datos")
            failed_competitions.append(code)
            continue
        
        valid_count = 0
        invalid_count = 0
        
        for team in api_data['teams']:
            team_id = team.get('id')
            name = team.get('name')
            
            if not team_id or not name:
                invalid_count += 1
                continue
            
            if team_id not in all_teams_data:
                all_teams_data[team_id] = {
                    'id': team_id,
                    'name': name,
                    'tla': team.get('tla', 'N/A')[:3],
                    'logo_url': team.get('crest', 'N/A')
                }
            
            team_competition_relations.append((team_id, code))
            valid_count += 1
        
        stats_by_competition[code] = valid_count
        
        log_msg = f"  └─ ✓ {valid_count} equipos"
        if invalid_count > 0:
            log_msg += f" ({invalid_count} inválidos)"
        logger.info(log_msg)
        
        if idx < len(competition_codes):
            time.sleep(config['api']['rate_limit_delay'])
    
    if not all_teams_data:
        logger.error("No hay equipos válidos")
        return {
            'success': False,
            'teams': 0,
            'relations': 0,
            'failed': failed_competitions
        }
    
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insertar equipos
        sql_teams = """
            INSERT INTO equipos (id, nombre, tla, logo_url)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                nombre = EXCLUDED.nombre,
                tla = EXCLUDED.tla,
                logo_url = EXCLUDED.logo_url,
                fecha_actualizacion = CURRENT_TIMESTAMP;
        """
        
        teams_to_insert = [
            (t['id'], t['name'], t['tla'], t['logo_url'])
            for t in all_teams_data.values()
        ]
        
        extras.execute_values(
            cursor,
            sql_teams,
            teams_to_insert,
            page_size=config['database']['batch_size']
        )
        
        teams_affected = cursor.rowcount
        logger.info(f"✓ {teams_affected} equipos procesados")
        
        # Insertar relaciones
        sql_relations = """
            INSERT INTO equipos_competiciones (equipo_id, competicion_id)
            VALUES %s
            ON CONFLICT (equipo_id, competicion_id) DO NOTHING;
        """
        
        extras.execute_values(
            cursor,
            sql_relations,
            team_competition_relations,
            page_size=config['database']['batch_size']
        )
        
        relations_affected = cursor.rowcount
        logger.info(f"✓ {relations_affected} relaciones nuevas")
        
        conn.commit()
        
        logger.info("")
        logger.info("Resumen por competición:")
        for code in sorted(stats_by_competition.keys()):
            logger.info(f"  • {code:>4}: {stats_by_competition[code]:>3} equipos")
        
        return {
            'success': True,
            'teams': len(all_teams_data),
            'teams_affected': teams_affected,
            'relations': len(team_competition_relations),
            'relations_affected': relations_affected,
            'failed': failed_competitions,
            'stats': stats_by_competition
        }
        
    except DatabaseError as e:
        logger.error(f"Error de base de datos: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {
            'success': False,
            'error': str(e)
        }
        
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {
            'success': False,
            'error': str(e)
        }
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== INGESTIÓN DE PARTIDOS ====================
def ingest_matches(config: Dict[str, Any], competition_codes: List[str]) -> Dict[str, Any]:
    """
    Ingesta partidos finalizados para las competiciones especificadas.
    
    Returns:
        Diccionario con estadísticas
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 3: INGESTIÓN DE PARTIDOS")
    logger.info("=" * 70)
    
    all_matches_data: Dict[int, Tuple] = {}
    failed_competitions = []
    stats_by_competition = {}
    
    @retry_on_failure(config['api']['max_retries'], config['api']['retry_delay'])
    def fetch_matches(code: str):
        return get_data(f"competitions/{code}/matches?status=FINISHED")
    
    for idx, code in enumerate(competition_codes, 1):
        logger.info(f"[{idx}/{len(competition_codes)}] Procesando: {code}")
        
        try:
            api_data = fetch_matches(code)
        except Exception as e:
            logger.error(f"  └─ ✗ Error: {e}")
            failed_competitions.append(code)
            continue
        
        if not api_data or 'matches' not in api_data:
            logger.warning(f"  └─ ✗ Sin datos")
            failed_competitions.append(code)
            continue
        
        valid_count = 0
        invalid_count = 0
        
        for match in api_data['matches']:
            validated = validate_match_data(match, code)
            if validated:
                match_id = validated[0]
                all_matches_data[match_id] = validated
                valid_count += 1
            else:
                invalid_count += 1
        
        stats_by_competition[code] = valid_count
        
        log_msg = f"  └─ ✓ {valid_count} partidos válidos"
        if invalid_count > 0:
            log_msg += f" ({invalid_count} inválidos)"
        logger.info(log_msg)
        
        if idx < len(competition_codes):
            time.sleep(config['api']['rate_limit_delay'])
    
    if not all_matches_data:
        logger.error("No hay partidos válidos para insertar")
        return {
            'success': False,
            'affected': 0,
            'failed': failed_competitions
        }
    
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar integridad referencial
        all_team_ids = set()
        for data in all_matches_data.values():
            all_team_ids.add(data[3])  # home
            all_team_ids.add(data[4])  # away
        
        cursor.execute(
            "SELECT id FROM equipos WHERE id = ANY(%s);",
            (list(all_team_ids),)
        )
        existing_teams = {row[0] for row in cursor.fetchall()}
        missing_teams = all_team_ids - existing_teams
        
        if missing_teams:
            logger.warning(
                f"⚠ {len(missing_teams)} equipos no existen en BD. "
                f"Filtrando partidos relacionados..."
            )
            original_count = len(all_matches_data)
            all_matches_data = {
                mid: data for mid, data in all_matches_data.items()
                if data[3] in existing_teams and data[4] in existing_teams
            }
            filtered = original_count - len(all_matches_data)
            logger.info(f"  └─ {filtered} partidos filtrados")
        
        if not all_matches_data:
            logger.error("No quedan partidos válidos después del filtrado")
            return {
                'success': False,
                'affected': 0,
                'failed': failed_competitions
            }
        
        sql = """
            INSERT INTO partidos
            (id, fecha_partido, competicion_id, equipo_local_id, equipo_visitante_id,
             goles_local, goles_visitante, resultado, estado)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                fecha_partido = EXCLUDED.fecha_partido,
                goles_local = EXCLUDED.goles_local,
                goles_visitante = EXCLUDED.goles_visitante,
                resultado = EXCLUDED.resultado,
                estado = EXCLUDED.estado;
        """
        
        extras.execute_values(
            cursor,
            sql,
            list(all_matches_data.values()),
            page_size=config['database']['batch_size']
        )
        
        affected = cursor.rowcount
        conn.commit()
        
        logger.info("")
        logger.info(f"✓ {affected} partidos procesados en BD")
        
        logger.info("")
        logger.info("Resumen por competición:")
        for code in sorted(stats_by_competition.keys()):
            logger.info(f"  • {code:>4}: {stats_by_competition[code]:>3} partidos")
        
        return {
            'success': True,
            'total': len(all_matches_data),
            'affected': affected,
            'failed': failed_competitions,
            'stats': stats_by_competition
        }
        
    except IntegrityError as e:
        logger.error(f"Error de integridad: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {
            'success': False,
            'affected': 0,
            'error': str(e)
        }
        
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {
            'success': False,
            'affected': 0,
            'error': str(e)
        }
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== INGESTIÓN DE POSICIONES ====================
def ingest_standings(config: Dict[str, Any], competition_codes: List[str]) -> Dict[str, Any]:
    """
    Ingesta tablas de posiciones actuales.
    
    Returns:
        Diccionario con estadísticas
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 4: INGESTIÓN DE POSICIONES")
    logger.info("=" * 70)
    
    all_standings_data = []
    failed_competitions = []
    stats_by_competition = {}
    
    @retry_on_failure(config['api']['max_retries'], config['api']['retry_delay'])
    def fetch_standings(code: str):
        return get_data(f"competitions/{code}/standings")
    
    for idx, code in enumerate(competition_codes, 1):
        logger.info(f"[{idx}/{len(competition_codes)}] Procesando: {code}")
        
        try:
            api_data = fetch_standings(code)
        except Exception as e:
            logger.error(f"  └─ ✗ Error: {e}")
            failed_competitions.append(code)
            continue
        
        if not api_data or 'standings' not in api_data or not api_data['standings']:
            logger.warning(f"  └─ ✗ Sin datos")
            failed_competitions.append(code)
            continue
        
        # Primera tabla = clasificación general
        table = api_data['standings'][0].get('table', [])
        
        valid_count = 0
        invalid_count = 0
        
        for row in table:
            validated = validate_standing_data(row, code)
            if validated:
                all_standings_data.append(validated)
                valid_count += 1
            else:
                invalid_count += 1
        
        stats_by_competition[code] = valid_count
        
        log_msg = f"  └─ ✓ {valid_count} posiciones"
        if invalid_count > 0:
            log_msg += f" ({invalid_count} inválidas)"
        logger.info(log_msg)
        
        if idx < len(competition_codes):
            time.sleep(config['api']['rate_limit_delay'])
    
    if not all_standings_data:
        logger.error("No hay posiciones válidas para insertar")
        return {
            'success': False,
            'affected': 0,
            'failed': failed_competitions
        }
    
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO posiciones
            (competicion_id, equipo_id, posicion, puntos, partidos_jugados,
             ganados, empatados, perdidos, goles_favor, goles_contra, diferencia_goles)
            VALUES %s
            ON CONFLICT (competicion_id, equipo_id, fecha_actualizacion)
            DO UPDATE SET
                posicion = EXCLUDED.posicion,
                puntos = EXCLUDED.puntos,
                partidos_jugados = EXCLUDED.partidos_jugados,
                ganados = EXCLUDED.ganados,
                empatados = EXCLUDED.empatados,
                perdidos = EXCLUDED.perdidos,
                goles_favor = EXCLUDED.goles_favor,
                goles_contra = EXCLUDED.goles_contra,
                diferencia_goles = EXCLUDED.diferencia_goles;
        """
        
        extras.execute_values(
            cursor,
            sql,
            all_standings_data,
            page_size=config['database']['batch_size']
        )
        
        affected = cursor.rowcount
        conn.commit()
        
        logger.info("")
        logger.info(f"✓ {affected} posiciones procesadas en BD")
        
        logger.info("")
        logger.info("Resumen por competición:")
        for code in sorted(stats_by_competition.keys()):
            logger.info(f"  • {code:>4}: {stats_by_competition[code]:>2} equipos")
        
        return {
            'success': True,
            'total': len(all_standings_data),
            'affected': affected,
            'failed': failed_competitions,
            'stats': stats_by_competition
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {
            'success': False,
            'affected': 0,
            'error': str(e)
        }
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== MAIN ====================
if __name__ == '__main__':
    start_time = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " PIPELINE DE INGESTIÓN DE DATOS v3.2 ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    # Cargar configuración
    config = load_config()
    leagues = config['competitions']['top_leagues']
    
    # Fase 1: Competiciones
    if not ingest_competitions(config):
        logger.error("Pipeline abortado: falló ingestión de competiciones")
        exit(1)
    
    # Fase 2: Equipos
    teams_result = ingest_teams_and_relations(config, leagues)
    if not teams_result.get('success', False):
        logger.error("Pipeline abortado: falló ingestión de equipos")
        exit(1)
    
    # Fase 3: Partidos
    matches_result = ingest_matches(config, leagues)
    
    # Fase 4: Posiciones
    standings_result = ingest_standings(config, leagues)
    
    # Resumen final
    elapsed = time.time() - start_time
    
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " RESUMEN FINAL ".center(68) + "║")
    logger.info("╠" + "═" * 68 + "╣")
    logger.info(f"║  Tiempo total: {elapsed:.2f}s".ljust(69) + "║")
    logger.info(f"║  Competiciones: ✓".ljust(69) + "║")
    logger.info(f"║  Equipos: {teams_result.get('teams_affected', 0)}".ljust(69) + "║")
    logger.info(f"║  Relaciones: {teams_result.get('relations_affected', 0)}".ljust(69) + "║")
    logger.info(f"║  Partidos: {matches_result.get('affected', 0)}".ljust(69) + "║")
    logger.info(f"║  Posiciones: {standings_result.get('affected', 0)}".ljust(69) + "║")
    
    # Verificar errores
    all_failed = set()
    if teams_result.get('failed'):
        all_failed.update(teams_result['failed'])
    if matches_result.get('failed'):
        all_failed.update(matches_result['failed'])
    if standings_result.get('failed'):
        all_failed.update(standings_result['failed'])
    
    if all_failed:
        logger.info(f"║  ⚠ Competiciones con errores: {len(all_failed)}".ljust(69) + "║")
        logger.info(f"║    {', '.join(sorted(all_failed))}".ljust(69) + "║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  Estado: COMPLETADO CON ADVERTENCIAS".ljust(69) + "║")
        logger.info("╚" + "═" * 68 + "╝")
        exit(1)
    else:
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  Estado: ✓ COMPLETADO EXITOSAMENTE".ljust(69) + "║")
        logger.info("╚" + "═" * 68 + "╝")
        exit(0)
