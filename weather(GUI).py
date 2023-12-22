#==================================================================================================================사용 패키지


import sys
from datetime import date, timedelta
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
import tkinter as tk
import tkinter.font as tkf
import os


#==================================================================================================================tkinter 사용


window = tk.Tk()

window.title("weather_ML")
window.geometry("380x300")

t_date1 = []
t_date2 = []
t_date3 = []

def btncmd():
    global t_date1, t_date2, t_date3
    t_date1 = entry1.get()
    t_date2 = entry2.get()
    t_date3 = entry3.get()
    label4.config(text='날짜입력! 준비완료!')
    btn2.config(fg='white', bg='red')

#처음 만들었던 weather.py 전체를 함수형태로 만듦
def forecasttt():
    label4.config(text='계산중...')

    weather = pd.read_csv(r'/절대경로를 집어 넣어야 실행 가능/WT.csv', index_col="DATE", encoding = "EUC-KR")


    weather = weather[['최저기온(°C)', 
                    '평균기온(°C)', 
                    '최고기온(°C)', 
                    '평균 상대습도(%)', 
                    '평균 증기압(hPa)', 
                    '평균 전운량(1/10)', 
                    '일강수량(mm)']].copy()

    weather.loc[weather['일강수량(mm)'] != weather['일강수량(mm)'], '일강수량(mm)'] = 0
    weather = weather.ffill()

    weather.index = pd.to_datetime(weather.index)

    weather["예상최저기온"] = weather.shift(-1)["최저기온(°C)"]
    weather["예상평균기온"] = weather.shift(-1)["평균기온(°C)"]
    weather["예상최고기온"] = weather.shift(-1)["최고기온(°C)"]
    weather["예상평균습도"] = weather.shift(-1)["평균 상대습도(%)"]
    weather["예상평균증기압"] = weather.shift(-1)["평균 증기압(hPa)"]
    weather["예상전운량"] = weather.shift(-1)["평균 전운량(1/10)"]
    weather["예상강수량"] = weather.shift(-1)["일강수량(mm)"]


    weather1 = weather.drop(["예상최저기온", 
                            "예상평균기온", 
                            "예상최고기온", 
                            "예상평균습도", 
                            "예상평균증기압", 
                            "예상전운량", 
                            "예상강수량"], axis=1)
    weather1 = weather1.drop(weather1.index[-1])
    weather1_t = weather1[['최저기온(°C)', '평균기온(°C)', '최고기온(°C)', 
                        '평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()
    weather1_tt = weather1_t
    weather1_r = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)', '일강수량(mm)']].copy()
    weather1_r_one = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)', '일강수량(mm)']].copy()

    weather1_r_one['일강수량(mm)'] = weather1_r_one['일강수량(mm)'] > 0
    X1 = weather1_r_one['일강수량(mm)'].map(lambda x : 1 if x else 0)
    Y1 = weather1_r_one.drop('일강수량(mm)', axis=1)
    weather1_r_one = pd.concat([X1, Y1], axis=1)

    weather1_rr = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()


    weather2 = pd.concat([weather["예상최저기온"], 
                        weather["예상평균기온"], 
                        weather["예상최고기온"], 
                        weather["예상평균습도"], 
                        weather["예상평균증기압"], 
                        weather["예상전운량"], 
                        weather["예상강수량"]], axis=1)
    weather2 = weather2.drop(weather2.index[-1])
    weather2_t = weather2[['예상최저기온', '예상평균기온', '예상최고기온', 
                        '예상평균습도', '예상평균증기압', '예상전운량']].copy()
    weather2_r = weather2[['예상평균습도', '예상평균증기압', '예상전운량', '예상강수량']].copy()
    weather2_r_one = weather2[['예상평균습도', '예상평균증기압', '예상전운량', '예상강수량']].copy()

    weather2_r_one['예상강수량'] = weather2_r_one['예상강수량'] > 0
    X2 = weather2_r_one['예상강수량'].map(lambda x : 1 if x else 0)
    Y2 = weather2_r_one.drop('예상강수량', axis=1)
    weather2_r_one = pd.concat([X2, Y2], axis=1)

    weather2_rr = weather2[['예상평균습도', '예상평균증기압', '예상전운량']].copy()


    weather3 = weather.drop(weather.index[-1])


    columns = [list(weather1_t.columns), list(weather2.columns), list(weather2_t.columns)]


    rr = Ridge(alpha=.1)

    def temp(weather1_t, weather2_t, model, start=3683, step=90, j=0, k=0):

        all_predictions = []

        columns2 = columns[k]
        column = columns2[j]
            
        for i in range(start, weather1_tt.shape[0], step):
            a = weather1_t.shape[0] - weather2_t.shape[0]
            weather1_t = weather1_t.iloc[:weather1_t.shape[0]-a,:]
            
            train1 = weather1_t.iloc[:i,:]
            train2 = weather2_t.iloc[:i,:]
            test1 = weather1_tt.iloc[i:(i+step),:]
            test2 = weather2_t.iloc[i:(i+step),:]
            
            model.fit(train1[weather1_t.columns], train2[column])

            preds = model.predict(test1[weather1_t.columns])
            preds = pd.Series(preds, index = test1.index)

            combined = pd.concat([test2[column], preds], axis=1)
            combined.columns = ["actual", column]

            all_predictions.append(combined)
        return pd.concat(all_predictions)


    timestamp_obj = weather1_t.index[-1]
    date_obj = timestamp_obj.to_pydatetime().date()

    start_date = date_obj
    end_date = date(int(t_date1),int(t_date2),int(t_date3))
    a = int((end_date - start_date).days)
    if a<=0:
        a=0

    b = weather.shape[0]


    for m in range(a):
        allpreds = []
        for n in range(6):
            predictions = temp(weather1_t, weather2_t, rr, j=n, k=2)
            predictions = predictions.drop('actual', axis=1)
            allpreds.append(predictions)
        allpreds = pd.concat(allpreds, axis=1)
        allpreds.columns = columns[0]

        weather1_index = list(weather1_tt.index)
        new = pd.to_datetime(weather1_tt.index[-1]) + timedelta(days=1)
        weather1_index.append(new)
        new_weather1_index = weather1_index
        
        weather1_tt.loc[b+m] = allpreds.loc[allpreds.index[-1]]
        weather1_tt.index = pd.to_datetime(new_weather1_index)


    weather1_rr = weather1_t[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()


    weatherf_t = []
    for n in range(6):
        w = temp(weather1_t, weather2_t, rr, j=n, k=1)
        w = w.drop('actual', axis=1)
        weatherf_t.append(w)
    weatherf_t = pd.concat(weatherf_t, axis=1)
    weatherf_t.columns = columns[2]


    weatherf_t[weather2_t.columns] = weatherf_t.shift(+1)[weather2_t.columns]
    weatherf_t = weatherf_t.drop(weatherf_t.index[0])

    weatherf_rr = weatherf_t[['예상평균습도', '예상평균증기압', '예상전운량']].copy()

    weather1_rr = weather1_rr.drop(weather1_rr.index[:3684])
    weather2_r_one = weather2_r_one.drop(weather2_r_one.index[:3684])


    lr = LogisticRegression()

    def rain_OX(weather1_rr, weather2_r_one, model, start=3683, step=90):

        all_predictions = []
            
        for i  in range(start, weatherf_rr.shape[0], step):
            a = weather1_rr.shape[0] - weather2_r_one.shape[0]
            weather1_rrr = weather1_rr.iloc[:weather1_rr.shape[0]-a,:]
            
            train1 = weather1_rrr.iloc[:i, :]
            train2 = weather2_r_one.iloc[:i, :]
            test = weather1_rr.iloc[i:(i+step),:]

            model.fit(train1[weather1_rr.columns], train2['예상강수량'])

            preds = model.predict(test[weather1_rr.columns])
            preds = pd.Series(preds, index = test.index)

            combined = pd.concat([test['평균 상대습도(%)'], preds], axis=1)
            combined.columns = ["actual", '예상강수량']

            all_predictions.append(combined)
        return pd.concat(all_predictions)


    rain_OX = rain_OX(weather1_rr, weather2_r_one, lr)
    rain_OX = rain_OX.drop('actual', axis=1)

    weatherf_t = weatherf_t.drop(weatherf_t.index[:3683])
    weatherff = pd.concat([weatherf_t[weatherf_t.columns], rain_OX[rain_OX.columns]], axis=1)

    weatherfff = weatherff[['예상강수량']].copy()
    weatherfff.columns = ['강수여부']

    weatherfff.index.name = 'DATE'
    target_date = t_date1+'-'+t_date2+'-'+t_date3
    t = weatherfff.query('DATE == @target_date')

    label4.config(text = t)
    

font=tkf.Font(family="맑은 고딕", size=20, slant="italic")

label1 = tk.Label(window, text='년', font=font)
label2 = tk.Label(window, text='월', font=font)
label3 = tk.Label(window, text='일', font=font)

entry1 = tk.Entry(window)
entry2 = tk.Entry(window)
entry3 = tk.Entry(window)

label1.grid(row=0, column=0, rowspan=2)
entry1.grid(row=0, column=6, padx=25, sticky='ew')
label2.grid(row=2, column=0, rowspan=2)
entry2.grid(row=2, column=6, padx=25, sticky='ew')
label3.grid(row=4, column=0, rowspan=2)
entry3.grid(row=4, column=6, padx=25, sticky='ew')

label4 = tk.Label(window, text='강수여부를 알고싶은 날짜를 입력해 주세요.')
label4.grid(row=6, column=3, rowspan=20, columnspan=20)

btn1 = tk.Button(window, text='DATE', padx=10, pady=10, command=btncmd)
btn1.grid(row=0, column=18, sticky='news', padx=3, pady=3)
btn2 = tk.Button(window, text='Forecast', padx=10, pady=10, command=forecasttt)
btn2.grid(row=4, column=18, sticky='news', padx=3, pady=3)

window.mainloop()