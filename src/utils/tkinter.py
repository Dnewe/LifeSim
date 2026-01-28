import tkinter as tk


class MenuSelector(tk.Menubutton):
    def __init__(self, parent, values, default=None, **kwargs):
        self.var = tk.Variable(value=default)
        super().__init__(
            parent,
            textvariable=self.var,
            relief=tk.RAISED,
            **kwargs
        )
        self.values = None
        self.menu = tk.Menu(self, tearoff=0)
        self.config(menu=self.menu)
        self.set_values(values, default)

    def set_values(self, values, default=None):
        if self.values == values:
            return
        self.values = values
        self.menu.delete(0, "end")

        for v in values:
            self.menu.add_radiobutton(
                label=str(v),
                value=v,
                variable=self.var
            )

        if default is not None:
            self.var.set(default)
        else:
            if self.get() not in values and len(values)>0:
                self.var.set(values[0])

    def get(self):
        return self.var.get()