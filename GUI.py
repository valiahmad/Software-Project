import sys
from PyQt5.QtWidgets import QApplication, QDialog, QLabel

def window():
    app = QApplication(sys.argv)
    win = QDialog()
    win.setGeometry(100,100,200,200)
    win.setWindowTitle('VALI AHMAD')

    lbl = QLabel()
    lbl.move(20,10)
    lbl.setText('Dash vali')
    lbl.adjustSize()




    win.show()
    sys.exit(app.exec_())

