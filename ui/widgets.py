"""
ChefMate Robot Assistant - Custom UI Widgets
Modern styled widgets for consistent UI design
"""

import tkinter as tk
from tkinter import ttk


class ModernButton(tk.Button):
    """Custom styled button with hover effects"""
    def __init__(self, parent, text="", command=None, bg='#2196F3', **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            font=('Segoe UI', 11, 'bold'),
            bg=bg,
            fg='white',
            activebackground='#1976D2',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            **kwargs
        )
        self.default_bg = bg
        self.hover_bg = self._darken_color(bg)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def _darken_color(self, hex_color):
        """Darken a hex color by 20%"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker = tuple(int(c * 0.8) for c in rgb)
        return f'#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}'

    def on_enter(self, e):
        self['background'] = self.hover_bg

    def on_leave(self, e):
        self['background'] = self.default_bg


class ScrollableFrame(tk.Frame):
    """A scrollable frame container"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, bg=kwargs.get('bg', '#f5f5f5'), highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=kwargs.get('bg', '#f5f5f5'))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self._update_scrollregion()
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Bind canvas resize to update frame width
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling when hovering over the frame
        self.bind_mousewheel()

    def _update_scrollregion(self):
        """Update the scroll region to encompass all content"""
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Update the width of the frame to match canvas"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def bind_mousewheel(self):
        """Bind mousewheel events to this frame and its children"""
        def _on_mousewheel(event):
            if self.canvas.winfo_height() < self.scrollable_frame.winfo_height():
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)
