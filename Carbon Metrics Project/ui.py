"""
Module responsible for displaying the plot results in an interactive Tkinter application.
"""

import tkinter as tk
import tkinter.font as fnt
from typing import Callable

import plot


class UserInterface:
    """
    The UserInterface class initializes the GUI and creates all the buttons and links to each
    specific function that the button does.
    """
    master: tk.Tk
    frame: tk.Frame
    text: tk.Label

    def __init__(self, master: tk.Tk) -> None:
        """Initializer for UserInterface."""

        # Initialize master window
        self.master = master
        self.frame = tk.Frame(self.master)
        self.master.geometry('1000x370')
        self.master.title('Carbon Metrics')

        # Background colour
        self.master['bg'] = '#89d0f0'

        # Welcoming text
        self.text = tk.Label(
            self.master,
            text="Welcome to Carbon Metrics",
            font=("Arial", 38),
            bg='#89d0f0',
            fg='black'
        )
        self.text.pack(pady=25)

        # Create frame
        self.frame = tk.Frame(self.master, bg='#89d0f0')
        self.frame.pack(pady=25)

        # Button Emissions By State
        self.create_button(
            text='Emissions By State',
            command=plot.bar_chart,
            column=0,
            row=0
        )

        # Button Data Used In Keras Model
        self.create_button(
            text='Data Used In Keras Model',
            command=plot.line_graph_one_line,
            column=1,
            row=0
        )

        # Button Predicted Verses Actual Emissions
        self.create_button(
            text='Predicted Verses Actual Emissions',
            command=plot.line_graph_predicted_vs_actual,
            column=0,
            row=1
        )

        # Button Difference of Prediction and Actual Emissions
        self.create_button(
            text='Difference of Prediction and Actual Emissons',
            command=plot.line_graph_predicted_minus_actual,
            column=1,
            row=1
        )

        # Button Predicted Verses Actual Emissions with model trained with normalized data
        self.create_button(
            text='Predicted Verses Actual Emissions (normalized)',
            command=predicted_vs_actual,
            column=0,
            row=2
        )

        # Button Prediction - Actual Emissions with model trained with normalized data
        self.create_button(
            text='Difference of Prediction and Actual Emissions (normalized)',
            command=predicted_minus_actual,
            column=1,
            row=2
        )

        # Pack everything together
        self.frame.pack()

    def create_button(self, text: str, command: Callable, row: int, column: int) -> None:
        """
        Modularly creates a button object and formats it according to the parameters.
        """
        button = tk.Button(
            self.frame,
            text=text,
            width=45,
            command=command,
            font=fnt.Font(size=12),
            activeforeground='red'
        )
        button.grid(row=row, column=column, padx=20, pady=10)


def predicted_minus_actual() -> None:
    """
    Calls the line_graph_predicted_vs_actual() function from plot.py
    """
    plot.line_graph_predicted_minus_actual(normalized=True)


def predicted_vs_actual() -> None:
    """
    Calls the line_graph_predicted_vs_actual() function from plot.py
    """
    plot.line_graph_predicted_vs_actual(normalized=True)


def main() -> None:
    """
    Main function that runs the UI.
    """
    window = tk.Tk()
    UserInterface(window)
    window.mainloop()


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['plot', 'tkinter', 'tkinter.font', 'typing'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts

    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
