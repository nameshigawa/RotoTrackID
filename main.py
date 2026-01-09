"""Application entry point for the Alpha tool GUI.

This module creates the Qt application, instantiates the `AlphaToolGUI`, and
starts the Qt event loop. Keep this file minimal so it can be used as the
standard entry point for packaging or running from the command line.
"""

import sys
from PySide6.QtWidgets import QApplication
from gui import AlphaToolGUI


def main():
	"""Create the QApplication, show the GUI, and run the event loop."""
	app = QApplication(sys.argv)
	gui = AlphaToolGUI()
	gui.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
