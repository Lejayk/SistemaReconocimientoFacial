"""
Sistema de Reconocimiento Facial con Análisis de Emociones
==========================================================
Punto de entrada de la aplicación. Inicializa la base de datos,
configura el logging y lanza la interfaz gráfica CustomTkinter.
"""

import logging
import sys

from src.database.db_manager import DatabaseManager
from src.gui.app import App


def _setup_logging() -> None:
    """Configurar el sistema de logging para la aplicación."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reducir verbosidad de librerías externas
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def main() -> int:
    """Inicializar todos los subsistemas y arrancar el loop de la GUI."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando Sistema de Reconocimiento Facial...")

    # 1. Inicializar / migrar la base de datos
    db = DatabaseManager()
    db.init_db()
    logger.info("Base de datos lista.")

    # 2. Lanzar la interfaz gráfica
    app = App(db_manager=db)
    app.mainloop()

    logger.info("Aplicación finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
