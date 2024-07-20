import tensorflow as tf
import numpy as np
import pandas as pd
from tkinter import messagebox
from tkinter import Tk, Label, Entry, Button, LabelFrame
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Đọc dữ liệu từ file CSV
predictions_lstm = pd.read_csv('/Users/maiphuong/Library/CloudStorage/OneDrive-ThuyloiUniversity/Learn/Python/DA_TTNT/KTM/predictions_lstm.csv')
Y_test = predictions_lstm['Actual'].values
bestPredLSTM = predictions_lstm['Predicted'].values

predictions_gwo_lstm = pd.read_csv('/Users/maiphuong/Library/CloudStorage/OneDrive-ThuyloiUniversity/Learn/Python/DA_TTNT/KTM/predictions_gwo_lstm.csv')
bestPredLSTM_GWO = predictions_gwo_lstm['Predicted'].values

predictions_pso_lstm = pd.read_csv('/Users/maiphuong/Library/CloudStorage/OneDrive-ThuyloiUniversity/Learn/Python/DA_TTNT/KTM/predictions_lstm_with_pso.csv')
bestPredLSTM_PSO = predictions_pso_lstm['Predicted'].values


# Các hàm show tỷ lệ dự đoán của các mô hình LSTM, LSTM_GWO, LSTM_PSO trên các độ đo: Precision, Recall, F1, Accuracy
def AboutLSTM():
    precision = precision_score(Y_test, bestPredLSTM, average='macro') * 100
    recall = recall_score(Y_test, bestPredLSTM, average='macro') * 100
    f1 = f1_score(Y_test, bestPredLSTM, average='macro') * 100
    accuracy = accuracy_score(Y_test, bestPredLSTM) * 100
    messagebox.showinfo("Tỉ lệ dự đoán đúng của LSTM", f"Precision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1: {f1:.2f}%\nAccuracy: {accuracy:.2f}%")

def AboutLSTM_GWO():
    precision = precision_score(Y_test, bestPredLSTM_GWO, average='macro') * 100
    recall = recall_score(Y_test, bestPredLSTM_GWO, average='macro') * 100
    f1 = f1_score(Y_test, bestPredLSTM_GWO, average='macro') * 100
    accuracy = accuracy_score(Y_test, bestPredLSTM_GWO) * 100
    messagebox.showinfo("Tỉ lệ dự đoán đúng của LSTM_GWO", f"Precision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1: {f1:.2f}%\nAccuracy: {accuracy:.2f}%")

def AboutLSTM_PSO():
    precision = precision_score(Y_test, bestPredLSTM_PSO, average='macro') * 100
    recall = recall_score(Y_test, bestPredLSTM_PSO, average='macro') * 100
    f1 = f1_score(Y_test, bestPredLSTM_PSO, average='macro') * 100
    accuracy = accuracy_score(Y_test, bestPredLSTM_PSO) * 100
    messagebox.showinfo("Tỉ lệ dự đoán đúng của LSTM_PSO", f"Precision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1: {f1:.2f}%\nAccuracy: {accuracy:.2f}%")

def PredictWithLSTM():
    try:
        ph = float(entryPH.get())
        hardness = float(entryHardness.get())
        solids = float(entrySolids.get())
        chloramines = float(entryChloramines.get())
        sulfate = float(entrySulfate.get())
        conductivity = float(entryConductivity.get())
        organic_carbon = float(entryOrganicCarbon.get())
        trihalomethanes = float(entryTrihalomethanes.get())
        turbidity = float(entryTurbidity.get())

        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_data_reshaped = input_data.reshape((-1, input_data.shape[1], 1))

        model = tf.keras.models.load_model('lstm_model.h5')
        prediction = model.predict(input_data_reshaped)
        prediction_label = 'Tốt' if prediction > 0.5 else 'Không tốt'
        lbPredictLSTM.config(text=prediction_label)
    except ValueError:
        messagebox.showinfo("Cảnh báo", "Vui lòng nhập đầy đủ và đúng định dạng thông tin để dự đoán")

def PredictWithLSTM_GWO():
    try:
        ph = float(entryPH.get())
        hardness = float(entryHardness.get())
        solids = float(entrySolids.get())
        chloramines = float(entryChloramines.get())
        sulfate = float(entrySulfate.get())
        conductivity = float(entryConductivity.get())
        organic_carbon = float(entryOrganicCarbon.get())
        trihalomethanes = float(entryTrihalomethanes.get())
        turbidity = float(entryTurbidity.get())

        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_data_reshaped = input_data.reshape((-1, input_data.shape[1], 1))

        model = tf.keras.models.load_model('lstm_gwo_model.h5')
        prediction = model.predict(input_data_reshaped)
        prediction_label = 'Tốt' if prediction > 0.5 else 'Không tốt'
        lbPredictLSTM_GWO.config(text=prediction_label)
    except ValueError:
        messagebox.showinfo("Cảnh báo", "Vui lòng nhập đầy đủ và đúng định dạng thông tin để dự đoán")

def PredictWithLSTM_PSO():
    try:
        ph = float(entryPH.get())
        hardness = float(entryHardness.get())
        solids = float(entrySolids.get())
        chloramines = float(entryChloramines.get())
        sulfate = float(entrySulfate.get())
        conductivity = float(entryConductivity.get())
        organic_carbon = float(entryOrganicCarbon.get())
        trihalomethanes = float(entryTrihalomethanes.get())
        turbidity = float(entryTurbidity.get())

        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_data_reshaped = input_data.reshape((-1, input_data.shape[1], 1))

        model = tf.keras.models.load_model('lstm_pso_model.h5')
        prediction = model.predict(input_data_reshaped)
        prediction_label = 'Tốt' if prediction > 0.5 else 'Không tốt'
        lbPredictLSTM_PSO.config(text=prediction_label)
    except ValueError:
        messagebox.showinfo("Cảnh báo", "Vui lòng nhập đầy đủ và đúng định dạng thông tin để dự đoán")

# Phần thiết kế form
FORM = Tk()

FORM.resizable(False, False)
FORM.geometry('800x585')
FORM.title("Dự đoán chất lượng nước")

lbSpace = Label(FORM, text="Thông tin để đưa ra dự đoán chất lượng nước", font=("Arial", 20), wraplength=550).grid(row=0, column=0, columnspan=2, sticky="ew")

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

# Sắp xếp các thuật toán trên cùng một hàng
groupLSTM = LabelFrame(FORM, text="Thuật toán LSTM")
groupLSTM.grid(column=0, row=10, pady=5, padx=5)
btnAboutLSTM = Button(groupLSTM, text="Tỉ lệ dự đoán đúng của LSTM", bg="#C7CBD1", command=AboutLSTM).grid(column=0, row=0, padx=5, pady=15)
btnPredictLSTM = Button(groupLSTM, text="Dự đoán với LSTM", bg="#C7CBD1", command=PredictWithLSTM).grid(column=0, row=1, pady=5)
lbLSTM = Label(groupLSTM, text="Kết quả dự đoán\n(Tốt / Không tốt)").grid(row=2, column=0)
lbPredictLSTM = Label(groupLSTM, text="---")
lbPredictLSTM.grid(row=3, column=0)

groupLSTM_GWO = LabelFrame(FORM, text="Thuật toán LSTM_GWO")
groupLSTM_GWO.grid(column=1, row=10, pady=5, padx=5)
btnAboutLSTM_GWO = Button(groupLSTM_GWO, text="Tỉ lệ dự đoán đúng của LSTM_GWO", bg="#C7CBD1", command=AboutLSTM_GWO).grid(column=0, row=0, padx=5, pady=15)
btnPredictLSTM_GWO = Button(groupLSTM_GWO, text="Dự đoán với LSTM_GWO", bg="#C7CBD1", command=PredictWithLSTM_GWO).grid(column=0, row=1, pady=5)
lbLSTM_GWO = Label(groupLSTM_GWO, text="Kết quả dự đoán\n(Tốt / Không tốt)").grid(row=2, column=0)
lbPredictLSTM_GWO = Label(groupLSTM_GWO, text="---")
lbPredictLSTM_GWO.grid(row=3, column=0)

groupLSTM_PSO = LabelFrame(FORM, text="Thuật toán LSTM_PSO")
groupLSTM_PSO.grid(column=2, row=10, pady=5, padx=5)
btnAboutLSTM_PSO = Button(groupLSTM_PSO, text="Tỉ lệ dự đoán đúng của LSTM_PSO", bg="#C7CBD1", command=AboutLSTM_PSO).grid(column=0, row=0, padx=5, pady=15)
btnPredictLSTM_PSO = Button(groupLSTM_PSO, text="Dự đoán với LSTM_PSO", bg="#C7CBD1", command=PredictWithLSTM_PSO).grid(column=0, row=1, pady=5)
lbLSTM_PSO = Label(groupLSTM_PSO, text="Kết quả dự đoán\n(Tốt / Không tốt)").grid(row=2, column=0)
lbPredictLSTM_PSO = Label(groupLSTM_PSO, text="---")
lbPredictLSTM_PSO.grid(row=3, column=0)

# Chạy form chính
FORM.mainloop()
