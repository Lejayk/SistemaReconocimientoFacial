"""
ReportsScreen
=============
Pantalla de analíticas y registro de auditoría.

Muestra:
* Tabla de eventos de detección recientes (paginada).
* Gráfico de barras de detecciones por emoción.
* Gráfico de barras de detecciones por persona.
* Gráfico de pie de emociones por persona específica.
* Estadísticas generales del sistema.
* Exportación a CSV.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING, List

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_TABLE_COLUMNS = ("ID", "Persona", "Emoción", "Confianza", "Fecha/Hora")
_PAGE_SIZE = 20

# Paleta de colores para gráficos
_CHART_COLORS = [
    "#3498db", "#2ecc71", "#e74c3c", "#f1c40f",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


class ReportsScreen(ctk.CTkFrame):
    """Pantalla de reportes y estadísticas.

    Parameters
    ----------
    master:
        Widget padre Tk/CTk.
    db_manager:
        Instancia compartida de :class:`~src.database.db_manager.DatabaseManager`.
    """

    def __init__(self, master, db_manager: "DatabaseManager") -> None:
        super().__init__(master)
        self.db_manager = db_manager
        self._offset = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- Encabezado con botones de acción ----
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(
            header,
            text="📊 Reportes y Estadísticas",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=15, pady=10)

        # Botón de exportación CSV
        ctk.CTkButton(
            header,
            text="📥 Exportar CSV",
            command=self._export_csv,
            width=130,
            fg_color="#27ae60",
            hover_color="#1e8449",
        ).pack(side="right", padx=5)

        ctk.CTkButton(
            header, text="🔄 Actualizar", command=self._refresh, width=120
        ).pack(side="right", padx=15)

        # ---- Pestañas ----
        self._tabs = ctk.CTkTabview(self)
        self._tabs.grid(row=1, column=0, sticky="nsew")

        self._tabs.add("📋 Eventos")
        self._tabs.add("📈 Estadísticas")
        self._tabs.add("😀 Por Emoción")
        self._tabs.add("👤 Por Persona")

        self._build_events_tab(self._tabs.tab("📋 Eventos"))
        self._build_stats_tab(self._tabs.tab("📈 Estadísticas"))
        self._build_emotion_chart_tab(self._tabs.tab("😀 Por Emoción"))
        self._build_person_chart_tab(self._tabs.tab("👤 Por Persona"))

        self._refresh()

    # ------------------------------------------------------------------
    # Pestaña de Eventos
    # ------------------------------------------------------------------

    def _build_events_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Frame scrollable como tabla simple
        self._table_frame = ctk.CTkScrollableFrame(parent)
        self._table_frame.grid(row=0, column=0, sticky="nsew")

        # Encabezados de columna
        for col, name in enumerate(_TABLE_COLUMNS):
            ctk.CTkLabel(
                self._table_frame,
                text=name,
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=col, padx=8, pady=4, sticky="w")

        # Controles de paginación
        nav = ctk.CTkFrame(parent)
        nav.grid(row=1, column=0, sticky="ew", pady=5)

        self._prev_btn = ctk.CTkButton(
            nav, text="◀ Anterior", command=self._prev_page, width=100
        )
        self._prev_btn.pack(side="left", padx=10)

        self._page_label = ctk.CTkLabel(nav, text="Página 1")
        self._page_label.pack(side="left", padx=10)

        self._next_btn = ctk.CTkButton(
            nav, text="Siguiente ▶", command=self._next_page, width=100
        )
        self._next_btn.pack(side="left", padx=10)

    def _populate_table(self, logs: List[dict]) -> None:
        """Llenar la tabla con los registros de detección."""
        # Eliminar filas existentes (mantener encabezado, fila 0)
        for widget in self._table_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()

        for row_idx, log in enumerate(logs, start=1):
            values = [
                str(log["id"]),
                log["person_name"] or "—",
                log["dominant_emotion"] or "—",
                f"{log['confidence']:.2f}" if log["confidence"] is not None else "—",
                str(log["detected_at"])[:19] if log["detected_at"] else "—",
            ]
            for col_idx, val in enumerate(values):
                ctk.CTkLabel(self._table_frame, text=val).grid(
                    row=row_idx, column=col_idx, padx=8, pady=2, sticky="w"
                )

    # ------------------------------------------------------------------
    # Pestaña de Estadísticas Generales
    # ------------------------------------------------------------------

    def _build_stats_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self._stats_frame = parent

    def _render_stats(self) -> None:
        """Renderizar estadísticas generales del sistema."""
        for w in self._stats_frame.winfo_children():
            w.destroy()

        try:
            stats = self.db_manager.get_general_stats()
        except Exception:
            ctk.CTkLabel(
                self._stats_frame, text="Error al cargar estadísticas."
            ).pack(expand=True)
            return

        # Contenedor principal con estilo de tarjetas
        container = ctk.CTkFrame(self._stats_frame, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=20, pady=20)
        container.grid_columnconfigure((0, 1), weight=1)

        cards = [
            ("📊 Total Detecciones", str(stats["total_detections"]), "#3498db"),
            ("👥 Personas Registradas", str(stats["total_persons"]), "#2ecc71"),
            ("👤 Persona Más Frecuente", stats["most_frequent_person"], "#e67e22"),
            ("🎭 Emoción Más Frecuente", stats["most_frequent_emotion"], "#9b59b6"),
        ]

        for i, (title, value, color) in enumerate(cards):
            row, col = divmod(i, 2)
            card = ctk.CTkFrame(container, corner_radius=12)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=13),
                text_color="gray70",
            ).pack(padx=15, pady=(15, 5))

            ctk.CTkLabel(
                card, text=value,
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=color,
            ).pack(padx=15, pady=(0, 15))

    # ------------------------------------------------------------------
    # Pestaña de gráfico por Emoción
    # ------------------------------------------------------------------

    def _build_emotion_chart_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self._emotion_chart_frame = parent

    def _render_emotion_chart(self) -> None:
        """Renderizar gráfico de barras de detecciones por emoción."""
        for w in self._emotion_chart_frame.winfo_children():
            w.destroy()

        summary = self.db_manager.get_emotion_summary()
        if not summary:
            ctk.CTkLabel(
                self._emotion_chart_frame,
                text="Sin datos de emociones aún.",
            ).pack(expand=True)
            return

        emotions = [s["emotion"] for s in summary]
        counts = [s["count"] for s in summary]

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3b3b3b")
        bars = ax.bar(
            emotions, counts,
            color=_CHART_COLORS[: len(emotions)],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title("Detecciones por Emoción", color="white", fontsize=14)
        ax.set_xlabel("Emoción", color="white")
        ax.set_ylabel("Cantidad", color="white")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

        # Agregar valor encima de cada barra
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center",
                color="white",
                fontsize=10,
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._emotion_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    # ------------------------------------------------------------------
    # Pestaña de gráfico por Persona
    # ------------------------------------------------------------------

    def _build_person_chart_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        # Controles de filtro
        filter_frame = ctk.CTkFrame(parent)
        filter_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(
            filter_frame, text="Filtrar por persona:"
        ).pack(side="left", padx=(10, 5))

        self._person_filter = ctk.CTkComboBox(
            filter_frame,
            values=["Todas las personas"],
            command=self._on_person_filter_changed,
            width=250,
        )
        self._person_filter.pack(side="left", padx=5)
        self._person_filter.set("Todas las personas")

        self._person_chart_frame = ctk.CTkFrame(parent)
        self._person_chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

    def _on_person_filter_changed(self, selection: str) -> None:
        """Callback cuando cambia el filtro de persona."""
        self._render_person_chart()

    def _render_person_chart(self) -> None:
        """Renderizar gráfico por persona (general o filtrado)."""
        for w in self._person_chart_frame.winfo_children():
            w.destroy()

        selection = self._person_filter.get()

        if selection == "Todas las personas":
            self._render_all_persons_chart()
        else:
            self._render_single_person_chart(selection)

    def _render_all_persons_chart(self) -> None:
        """Gráfico horizontal de detecciones por persona."""
        stats = self.db_manager.get_person_detection_stats()
        if not stats:
            ctk.CTkLabel(
                self._person_chart_frame,
                text="Sin datos de personas aún.",
            ).pack(expand=True)
            return

        names = [s["person_name"] for s in stats]
        values = [s["count"] for s in stats]

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3b3b3b")
        bars = ax.barh(
            names, values,
            color=_CHART_COLORS[: len(names)],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title("Detecciones por Persona", color="white", fontsize=14)
        ax.set_xlabel("Cantidad", color="white")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

        # Valor al lado de cada barra
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                color="white",
                fontsize=10,
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._person_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def _render_single_person_chart(self, person_name: str) -> None:
        """Gráfico de pie de distribución de emociones para una persona."""
        # Buscar el person_id
        people = self.db_manager.get_all_people()
        person_id = None
        for p in people:
            full_name = f"{p['name']} {p.get('apellido', '')}".strip()
            if full_name == person_name:
                person_id = p["id"]
                break

        if person_id is None:
            # Intentar búsqueda por nombre simple
            for p in people:
                if p["name"] == person_name:
                    person_id = p["id"]
                    break

        if person_id is None:
            ctk.CTkLabel(
                self._person_chart_frame,
                text=f"No se encontraron datos para '{person_name}'.",
            ).pack(expand=True)
            return

        emotions = self.db_manager.get_emotions_by_person(person_id)
        if not emotions:
            ctk.CTkLabel(
                self._person_chart_frame,
                text=f"Sin datos de emociones para '{person_name}'.",
            ).pack(expand=True)
            return

        labels = [e["emotion"] for e in emotions]
        sizes = [e["count"] for e in emotions]

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2b2b2b")

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=_CHART_COLORS[: len(labels)],
            textprops={"color": "white", "fontsize": 10},
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")

        ax.set_title(
            f"Emociones de {person_name}",
            color="white",
            fontsize=14,
        )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._person_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def _update_person_filter(self) -> None:
        """Actualizar las opciones del combo de filtro de personas."""
        people = self.db_manager.get_all_people()
        names = ["Todas las personas"]
        for p in people:
            full_name = f"{p['name']} {p.get('apellido', '')}".strip()
            if full_name not in names:
                names.append(full_name)

        # También agregar nombres del log de detecciones
        stats = self.db_manager.get_person_detection_stats()
        for s in stats:
            if s["person_name"] and s["person_name"] not in names:
                names.append(s["person_name"])

        self._person_filter.configure(values=names)

    # ------------------------------------------------------------------
    # Exportación CSV
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        """Abrir diálogo para exportar historial de detecciones a CSV."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Archivo CSV", "*.csv"), ("Todos los archivos", "*.*")],
            title="Exportar historial de detecciones",
            initialfile="detecciones_export.csv",
        )
        if not filepath:
            return

        try:
            count = self.db_manager.export_logs_csv(filepath)
            # Mostrar confirmación temporal
            self._show_export_message(
                f"✅ {count} registros exportados exitosamente a:\n{os.path.basename(filepath)}"
            )
        except Exception as exc:
            self._show_export_message(f"❌ Error al exportar: {exc}")

    def _show_export_message(self, message: str) -> None:
        """Mostrar un mensaje temporal de exportación."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Exportación CSV")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog, text=message, wraplength=350,
            font=ctk.CTkFont(size=13),
        ).pack(expand=True, padx=20, pady=20)

        ctk.CTkButton(
            dialog, text="Aceptar", command=dialog.destroy, width=100
        ).pack(pady=(0, 15))

    # ------------------------------------------------------------------
    # Paginación
    # ------------------------------------------------------------------

    def _prev_page(self) -> None:
        if self._offset >= _PAGE_SIZE:
            self._offset -= _PAGE_SIZE
            self._refresh_table()

    def _next_page(self) -> None:
        self._offset += _PAGE_SIZE
        self._refresh_table()

    # ------------------------------------------------------------------
    # Actualización de datos
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        """Actualizar la tabla de eventos con paginación."""
        logs = self.db_manager.get_detection_logs(
            limit=_PAGE_SIZE, offset=self._offset
        )
        self._populate_table(logs)
        page = self._offset // _PAGE_SIZE + 1
        self._page_label.configure(text=f"Página {page}")
        self._prev_btn.configure(
            state="normal" if self._offset > 0 else "disabled"
        )
        self._next_btn.configure(
            state="normal" if len(logs) == _PAGE_SIZE else "disabled"
        )

    def _refresh(self) -> None:
        """Refrescar todos los datos: tabla, gráficos y estadísticas."""
        self._offset = 0
        self._refresh_table()
        self._render_stats()
        self._render_emotion_chart()
        self._update_person_filter()
        self._render_person_chart()
