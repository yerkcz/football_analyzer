# Contexto del Proyecto: Football Analyzer (v2)

## Fase 1: Cimiento Profesional (Completada)
- Estructura del Proyecto: `src`, `data`, `notebooks`, `saved_models`.
- Control de Versiones: Repositorio Git sincronizado con GitHub.
- Entorno Virtual: `venv` con dependencias en `requirements.txt`.

## Fase 2: Base de Datos Profesional (Completada)
- **Plataforma:** Proyecto `football-analyzer-pro` creado en Supabase.
- **Esquema SQL:** Diseñado y ejecutado con enfoque en escalabilidad.
  - Tablas creadas: `competiciones`, `equipos`, `partidos`, `posiciones`.
  - El diseño incluye claves foráneas para integridad referencial, restricciones `UNIQUE` y `CHECK` para calidad de datos.
  - El esquema ya prevé la expansión futura a estadísticas de jugadores (tablas `jugadores` y `estadisticas_partido_jugador` diseñadas y comentadas).
- **Gestión de Secretos:** Archivo `.env` creado para almacenar la `DATABASE_URI`. Este archivo está correctamente ignorado por Git.
- **Estado:** Base de datos lista y esperando datos. Conexión segura configurada.
