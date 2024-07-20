#!/usr/bin/env python3

from tkinter import Tk, Label, Entry, Button, LabelFrame
import tensorflow as tf
import numpy as np

# Các hàm để xử lý sự kiện
def AboutLSTM():
	# Giả định chúng ta đã lưu mô hình LSTM và tải mô hình này
	model = tf.keras.models.load_model('lstm_model.h5')
	# Hiển thị thông tin về mô hình LSTM (ví dụ: độ chính xác)
	accuracy = 0.85  # giả định độ chính xác của mô hình
	result = f"Độ chính xác của LSTM: {accuracy*100:.2f}%"
	lbPredictLSTM.config(text=result)
	
def PredictWithLSTM():
	# Thu thập dữ liệu từ form
	ph = float(entryPH.get())
	hardness = float(entryHardness.get())
	solids = float(entrySolids.get())
	chloramines = float(entryChloramines.get())
	sulfate = float(entrySulfate.get())
	conductivity = float(entryConductivity.get())
	organic_carbon = float(entryOrganicCarbon.get())
	trihalomethanes = float(entryTrihalomethanes.get())
	turbidity = float(entryTurbidity.get())
	
	# Tạo mảng đầu vào cho mô hình
	input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
	input_data_reshaped = input_data.reshape((-1, input_data.shape[1], 1))
	
	# Tải mô hình và dự đoán
	model = tf.keras.models.load_model('lstm_model.h5')
	prediction = model.predict(input_data_reshaped)
	prediction_label = 'Đạt chuẩn' if prediction > 0.5 else 'Không đạt chuẩn'
	lbPredictLSTM.config(text=prediction_label)
	
# Phần thiết kế form
FORM = Tk()

# Tắt chức năng thay đổi kích thước của form
FORM.resizable(False, False)

# Đặt kích thước cho form
FORM.geometry('470x585')

# Đặt tên cho form
FORM.title("Dự đoán chất lượng nước")

# Các đối tượng được dùng trong form: Label, Entry, Button, LabelFrame (Group)
lbSpace = Label(FORM, text="Thông tin để đưa ra dự đoán chất lượng nước", font=("Arial", 20), wraplength=450).grid(row=0, column=0, columnspan=2, sticky="ew")

lbPH = Label(FORM, text="pH:").grid(row=1, column=0, pady=5, sticky="e")
entryPH = Entry(FORM)
entryPH.grid(row=1, column=1, pady=5, sticky="we")

lbHardness = Label(FORM, text="Hardness:").grid(row=2, column=0, pady=5, sticky="e")
entryHardness = Entry(FORM)
entryHardness.grid(row=2, column=1, pady=5, sticky="we")

lbSolids = Label(FORM, text="Solids:").grid(row=3, column=0, pady=5, sticky="e")
entrySolids = Entry(FORM)
entrySolids.grid(row=3, column=1, pady=5, sticky="we")

lbChloramines = Label(FORM, text="Chloramines:").grid(row=4, column=0, pady=5, sticky="e")
entryChloramines = Entry(FORM)
entryChloramines.grid(row=4, column=1, pady=5, sticky="we")

lbSulfate = Label(FORM, text="Sulfate:").grid(row=5, column=0, pady=5, sticky="e")
entrySulfate = Entry(FORM)
entrySulfate.grid(row=5, column=1, pady=5, sticky="we")

lbConductivity = Label(FORM, text="Conductivity:").grid(row=6, column=0, pady=5, sticky="e")
entryConductivity = Entry(FORM)
entryConductivity.grid(row=6, column=1, pady=5, sticky="we")

lbOrganicCarbon = Label(FORM, text="Organic Carbon:").grid(row=7, column=0, pady=5, sticky="e")
entryOrganicCarbon = Entry(FORM)
entryOrganicCarbon.grid(row=7, column=1, pady=5, sticky="we")

lbTrihalomethanes = Label(FORM, text="Trihalomethanes:").grid(row=8, column=0, pady=5, sticky="e")
entryTrihalomethanes = Entry(FORM)
entryTrihalomethanes.grid(row=8, column=1, pady=5, sticky="we")

lbTurbidity = Label(FORM, text="Turbidity:").grid(row=9, column=0, pady=5, sticky="e")
entryTurbidity = Entry(FORM)
entryTurbidity.grid(row=9, column=1, pady=5, sticky="we")

# Nhóm LSTM
groupLSTM = LabelFrame(FORM, text="Mô hình LSTM")
groupLSTM.grid(column=0, row=10, columnspan=2, pady=5)
btnAboutLSTM = Button(groupLSTM, text="Tỉ lệ dự đoán đúng của LSTM", bg="#C7CBD1", command=AboutLSTM).grid(column=0, row=0, padx=5, pady=15)
btnPredictLSTM = Button(groupLSTM, text="Dự đoán với LSTM", bg="#C7CBD1", command=PredictWithLSTM).grid(column=0, row=1, pady=5)
lbLSTM = Label(groupLSTM, text="Kết quả dự đoán\n(Đạt chuẩn / Không đạt chuẩn)").grid(row=2, column=0)
lbPredictLSTM = Label(groupLSTM, text="---")
lbPredictLSTM.grid(row=3, column=0)

FORM.mainloop()
