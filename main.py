"""
Sistema de Reconocimiento Facial con Análisis de Emociones
==========================================================
Entry point of the application. Bootstraps the database, loads
configuration and launches the CustomTkinter GUI.
"""

import sys
from src.database.db_manager import DatabaseManager
from src.gui.app import App


def main() -> int:
    """Initialise all subsystems and start the GUI event loop."""
    # 1. Initialise / migrate the database
    db = DatabaseManager()
    db.init_db()

    # 2. Launch the GUI
    app = App(db_manager=db)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
