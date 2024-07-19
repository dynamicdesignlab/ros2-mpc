""" Common dialogs used for plotting

This module contains a collection of common dialogs used for plotting.

"""

import gi

from pathlib import Path
from matplotlib.backends import backend_gtk3

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402 - import must be under previous line


class NavigationToolbarNoCoordinates(backend_gtk3.NavigationToolbar2GTK3):
    """NavigationToolbar2TkNoCoordinates

    This custom toolbar class removes the normal coordinate
    readout of the default toolbar

    """

    def set_message(self, s):
        pass


def select_file_dialog(default_dir: Path, label="All Files", filt_pattern="*"):
    dialog = Gtk.FileChooserDialog(
        title="Please choose a file", action=Gtk.FileChooserAction.OPEN
    )
    dialog.add_buttons(
        Gtk.STOCK_CANCEL,
        Gtk.ResponseType.CANCEL,
        Gtk.STOCK_OPEN,
        Gtk.ResponseType.OK,
    )
    dialog.set_current_folder(str(default_dir))
    filter_obj = Gtk.FileFilter()
    filter_obj.set_name(label)
    filter_obj.add_pattern(filt_pattern)
    dialog.add_filter(filter_obj)

    response = dialog.run()
    if response == Gtk.ResponseType.CANCEL:
        dialog.destroy()
        exit()

    file_path = Path(dialog.get_filename())
    dialog.destroy()
    while Gtk.events_pending():
        Gtk.main_iteration()

    return file_path
