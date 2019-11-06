# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LRP.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from MatplotlibWidget_one import MatplotlibWidget_one
from MatplotlibWidget_many import MatplotlibWidget_many

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        #按钮
        self.horizontalLayoutWidget     = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(920, 520, 160, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout           = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.pushButton_ok              = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_ok.setObjectName("pushButton_ok")
        
        self.horizontalLayout.addWidget(self.pushButton_ok)
        
        self.pushButton_pass            = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_pass.setObjectName("pushButton_pass")
        
        self.horizontalLayout.addWidget(self.pushButton_pass)
        
        self.pushButton_save            = QtWidgets.QPushButton(Dialog)
        self.pushButton_save.setGeometry(QtCore.QRect(780, 730, 75, 25))
        self.pushButton_save.setObjectName("pushButton_save")
        
        self.pushButton_close           = QtWidgets.QPushButton(Dialog)
        self.pushButton_close.setGeometry(QtCore.QRect(1100, 770, 75, 25))
        self.pushButton_close.setObjectName("pushButton_close")
        
        self.pushButton_init            = QtWidgets.QPushButton(Dialog)
        self.pushButton_init.setGeometry(QtCore.QRect(1020, 770, 75, 25))
        self.pushButton_init.setObjectName("pushButton_init")
        #文本框
        self.QlineEdit_save = QtWidgets.QLineEdit(Dialog)
        self.QlineEdit_save.setGeometry(QtCore.QRect(860, 730, 200, 25))
        self.QlineEdit_save.setObjectName("QlineEdit_save")
        
        self.Qlable_time=QtWidgets.QLabel(Dialog)
        self.Qlable_time.setGeometry(QtCore.QRect(780, 600, 75, 25))
        self.Qlable_time.setObjectName("Qlable_time")
        self.Qlable_time.setText('Time')
        self.QlineEdit_time = QtWidgets.QLineEdit(Dialog)
        self.QlineEdit_time.setGeometry(QtCore.QRect(860, 600, 100, 25))
        self.QlineEdit_time.setObjectName("QlineEdit_time")

        self.Qlable_bias=QtWidgets.QLabel(Dialog)
        self.Qlable_bias.setGeometry(QtCore.QRect(780, 630, 75, 25))
        self.Qlable_bias.setObjectName("Qlable_bias")
        self.Qlable_bias.setText('Bias')
        self.QlineEdit_bias = QtWidgets.QLineEdit(Dialog)
        self.QlineEdit_bias.setGeometry(QtCore.QRect(860, 630, 100, 25))
        self.QlineEdit_bias.setObjectName("QlineEdit_bias")

        self.Qlable_idx_start=QtWidgets.QLabel(Dialog)
        self.Qlable_idx_start.setGeometry(QtCore.QRect(780, 660, 75, 25))
        self.Qlable_idx_start.setObjectName("Qlable_idx_start")
        self.Qlable_idx_start.setText('Idx_start')
        self.QlineEdit_idx_start = QtWidgets.QLineEdit(Dialog)
        self.QlineEdit_idx_start.setGeometry(QtCore.QRect(860, 660, 100, 25))
        self.QlineEdit_idx_start.setObjectName("QlineEdit_idx_start")
        #计数
        self.spinBox_target_label = QtWidgets.QSpinBox(Dialog)
        self.spinBox_target_label.setGeometry(QtCore.QRect(980, 560, 42, 22))
        self.spinBox_target_label.setObjectName("spinBox_target_label")
        #画图类
        self.many_ = MatplotlibWidget_many(Dialog)
        self.many_.setGeometry(QtCore.QRect(0, 0, 750, 800))
        self.many_.setObjectName("many_")
        
        self.one_ = MatplotlibWidget_one(parent=Dialog)
        self.one_.setGeometry(QtCore.QRect(800, 0, 400, 500))
        self.one_.setObjectName("one_")

        self.retranslateUi(Dialog)
        #信号连接
        self.pushButton_ok.clicked.connect(lambda:self.ok())
        self.pushButton_pass.clicked.connect(lambda:self.pass_())
        self.pushButton_save.clicked.connect(lambda:self.save())
        self.pushButton_close.clicked.connect(Dialog.close)
        self.pushButton_init.clicked.connect(lambda:self.init_MatplotlibWidget_one())
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_ok.setText(_translate("Dialog", "OK"))
        self.pushButton_pass.setText(_translate("Dialog", "pass"))
        self.pushButton_save.setText(_translate("Dialog", "SAVE"))
        self.pushButton_close.setText(_translate("Dialog", "close"))
        self.pushButton_init.setText(_translate("Dialog", "init"))
    
    @pyqtSlot()
    def save(self):
        self.many_.save(self.QlineEdit_save.text())
        
    @pyqtSlot() 
    def ok(self):
        Rx_sample_mean=self.one_.send()
        self.many_.get_Rx_sample_mean(Rx_sample_mean)
        self.many_.lrpplot(Rx_sample_mean)
        self.one_.pltlrp(self.spinBox_target_label.value())
#          print('ooooookkk')
          
    @pyqtSlot()      
    def pass_(self):
        self.one_.pltlrp(self.spinBox_target_label.value())
#          print('OKKKKK')
    
    @pyqtSlot()
    def init_MatplotlibWidget_one(self):
        self.one_.initplotUi(idx_start=int(self.QlineEdit_idx_start.text()),time=self.QlineEdit_time.text(),bias=self.QlineEdit_bias.text())
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Dialog()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
