#==================================================================================================================사용 패키지



import sys
from datetime import date, timedelta
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression



#====================================================================================================================외부 입력



t_date1 = sys.argv[1]
t_date2 = sys.argv[2]
t_date3 = sys.argv[3]



#=================================================================================================================데이터 전처리



#불러오기
weather = pd.read_csv("WT.csv", index_col="DATE", encoding = "EUC-KR")

#사용할 column만 때서 'weather' dataframe만들기
weather = weather[['최저기온(°C)', 
                   '평균기온(°C)', 
                   '최고기온(°C)', 
                   '평균 상대습도(%)', 
                   '평균 증기압(hPa)', 
                   '평균 전운량(1/10)', 
                   '일강수량(mm)']].copy()

#'weather' dataframe에서 nan값 제거
weather.loc[weather['일강수량(mm)'] != weather['일강수량(mm)'], '일강수량(mm)'] = 0
weather = weather.ffill()

#'weather' dataframe의 index를 날짜형으로 변환
weather.index = pd.to_datetime(weather.index)

#'weather' dataframe의 자료를 하루 뒤로 물림
weather["예상최저기온"] = weather.shift(-1)["최저기온(°C)"]
weather["예상평균기온"] = weather.shift(-1)["평균기온(°C)"]
weather["예상최고기온"] = weather.shift(-1)["최고기온(°C)"]
weather["예상평균습도"] = weather.shift(-1)["평균 상대습도(%)"]
weather["예상평균증기압"] = weather.shift(-1)["평균 증기압(hPa)"]
weather["예상전운량"] = weather.shift(-1)["평균 전운량(1/10)"]
weather["예상강수량"] = weather.shift(-1)["일강수량(mm)"]


#---------------------------------------------------------------------------------------------------------------------------


#'weather' dataframe에서 뒤로 물린 자료를 제거한 'weather1' dataframe 생성(즉 'weather' dataframe과 동일)
weather1 = weather.drop(["예상최저기온", 
                        "예상평균기온", 
                        "예상최고기온", 
                        "예상평균습도", 
                        "예상평균증기압", 
                        "예상전운량", 
                        "예상강수량"], axis=1)

#'weather2' dataframe과 index를 맞춤
weather1 = weather1.drop(weather1.index[-1])

#온도(t)를 예측하기 위해 상관성이 떨어지는 "일강수량(mm)" column을 제거한 'weather1_t' dataframe
weather1_t = weather1[['최저기온(°C)', '평균기온(°C)', '최고기온(°C)', 
                       '평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()

#중간에 오류가 생겨 'weather1_t' dataframe과 동일한 dataframe을 하나 더듦
weather1_tt = weather1_t

#"강수여부"(r)를 예측하기 위해 '일강수량(mm)' column과 상관성이 높은 자료들만 모음
weather1_r = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)', '일강수량(mm)']].copy()

#"강수여부"를 예측하기 위해 '일강수량(mm)' 값들을 binary(one)로 바꾼 'weather1_r_one' dataframe(1:비가 옴, 0:비가 오지 않음)
weather1_r_one = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)', '일강수량(mm)']].copy()

#'weather1_r_one'에서 '일강수량(mm)' 값들을 binary로 바꿈
weather1_r_one['일강수량(mm)'] = weather1_r_one['일강수량(mm)'] > 0
X1 = weather1_r_one['일강수량(mm)'].map(lambda x : 1 if x else 0)
Y1 = weather1_r_one.drop('일강수량(mm)', axis=1)
weather1_r_one = pd.concat([X1, Y1], axis=1)

#"강수여부"를 학습시키기 위해 '일강수량(mm)'과 상관성이 높으면서 자신은 제외한 column들만 모은 'weather1_rr' dataframe
weather1_rr = weather1[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()


#---------------------------------------------------------------------------------------------------------------------------


#'weather' dataframe에서 뒤로 물린 자료만 모은 'weather2' dataframe 생성
weather2 = pd.concat([weather["예상최저기온"], 
                      weather["예상평균기온"], 
                      weather["예상최고기온"], 
                      weather["예상평균습도"], 
                      weather["예상평균증기압"], 
                      weather["예상전운량"], 
                      weather["예상강수량"]], axis=1)

#'weather2' dataframe에서 뒤로 물려서 생긴 nan값이 있는 index 제거
weather2 = weather2.drop(weather2.index[-1])

#'weather2' dataframe에서 기온(t) 관련 학습 시 y값으로 들어갈 예상 값들을 모은 'weather2_t' dataframe (weather1_t의 예상)
weather2_t = weather2[['예상최저기온', '예상평균기온', '예상최고기온', 
                       '예상평균습도', '예상평균증기압', '예상전운량']].copy()

#'weather2' dataframe에서 강수여부(r) 관련 학습 시 필요한 값들을 모은 'weather2_r' dataframe (weather1_r의 예상)
weather2_r = weather2[['예상평균습도', '예상평균증기압', '예상전운량', '예상강수량']].copy()

#'weather2_r' dataframe에서 '예상강수량'값을 binary로 바꾼 'weather2_r_one' dataframe (weather1_r의 예상)
weather2_r_one = weather2[['예상평균습도', '예상평균증기압', '예상전운량', '예상강수량']].copy()

#'weather2_r_one'에서 '일강수량(mm)' 값들을 binary로 바꿈
weather2_r_one['예상강수량'] = weather2_r_one['예상강수량'] > 0
X2 = weather2_r_one['예상강수량'].map(lambda x : 1 if x else 0)
Y2 = weather2_r_one.drop('예상강수량', axis=1)
weather2_r_one = pd.concat([X2, Y2], axis=1)

#weather1_rr의 예상
weather2_rr = weather2[['예상평균습도', '예상평균증기압', '예상전운량']].copy()


#---------------------------------------------------------------------------------------------------------------------------


#weather1과 weather2값이 같이 있는 'weather3' dataframe
weather3 = weather.drop(weather.index[-1])



#======================================================================================================주어진 데이터를 이용한 학습



#중간에 학습시 필요한 자료만을 고르기 위해 특정 dataframe의 column들을 list형식으로 'columns'리스트에 저장
columns = [list(weather1_t.columns), list(weather2.columns), list(weather2_t.columns)]


#---------------------------------------------------------------------------------------------------------------------------


#온도를 예측하기 위한 Ridge regression
rr = Ridge(alpha=.1)

#___________________________________________________________________________temp 컴포넌트
#온도 예측
def temp(weather1_t, weather2_t, model, start=3683, step=90, j=0, k=0):

    all_predictions = []

    columns2 = columns[k]
    column = columns2[j]
        
    for i in range(start, weather1_tt.shape[0], step):
        
#예측을 위한 학습 시 오류가 나지 않도록 weather1_t의 index 수와 weather2_t의 index수를 맞춰줌
        a = weather1_t.shape[0] - weather2_t.shape[0]
        weather1_t = weather1_t.iloc[:weather1_t.shape[0]-a,:]
        
        train1 = weather1_t.iloc[:i,:]
        train2 = weather2_t.iloc[:i,:]
        test1 = weather1_tt.iloc[i:(i+step),:]
        test2 = weather2_t.iloc[i:(i+step),:]
        
#weather1_t의 값과 weather2_t의 값의 상관성을 훈련
        model.fit(train1[weather1_t.columns], train2[column])

#weather1_tt(=weather1_t)의 값으로 예측 
        preds = model.predict(test1[weather1_t.columns])
        preds = pd.Series(preds, index = test1.index)

        combined = pd.concat([test2[column], preds], axis=1)
        combined.columns = ["actual", column]

        all_predictions.append(combined)
    return pd.concat(all_predictions)
#__________________________________________________________________________________________

#weather1_t의 index를 더하기 위해 날짜형으로 만듦
timestamp_obj = weather1_t.index[-1]
date_obj = timestamp_obj.to_pydatetime().date()

#원하는 날짜와 입력된 데이터의 최근 날짜 사이의 차를 계산한 값을 가지는 'a'
start_date = date_obj
end_date = date(int(t_date1),int(t_date2),int(t_date3))
a = int((end_date - start_date).days)

#혹시 원하는 날짜가 입력된 데이터에 있는 날짜일 경우 오류를 방지함
if a<=0:
    a=0

b = weather.shape[0]

#예측한 값을 다시 예측을 위한 훈련 자료의 마지막 index로 옮겨 붙이는 과정
for m in range(a):
    allpreds = []
    
    #range(6)인 이유?
    #예측을 위한 훈련에 'weather1_t' dataframe의 모든 column들이 사용되므로 temp 컴포넌트를 이용해
    #'기온'뿐만이 아니라 다른 column들에 대해서도 예측 값을 구함
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

#weather1_rr의 값을 업데이트
weather1_rr = weather1_t[['평균 상대습도(%)', '평균 증기압(hPa)', '평균 전운량(1/10)']].copy()

#기온 예측 최종값
weatherf_t = []
for n in range(6):
    w = temp(weather1_t, weather2_t, rr, j=n, k=1)
    w = w.drop('actual', axis=1)
    weatherf_t.append(w)
weatherf_t = pd.concat(weatherf_t, axis=1)
weatherf_t.columns = columns[2]

#예측값으로 한칸 뒤로 물렸으므로 날짜를 맞추기 위해 다시 앞으로 당김
weatherf_t[weather2_t.columns] = weatherf_t.shift(+1)[weather2_t.columns]
weatherf_t = weatherf_t.drop(weatherf_t.index[0])


#---------------------------------------------------------------------------------------------------------------------------


#강수여부 측정을 위한 자료 업데이트
weatherf_rr = weatherf_t[['예상평균습도', '예상평균증기압', '예상전운량']].copy()

#학습 시킬때 자료 길이가 맞지 않아 오류가 생겨 자료 길이를 맞춰줌
weather1_rr = weather1_rr.drop(weather1_rr.index[:3684])
weather2_r_one = weather2_r_one.drop(weather2_r_one.index[:3684])


#강수여부 예측을 위한 Logistic Regression
lr = LogisticRegression()

#____________________________________________________________________________rain_OX 컴포넌트
#강수여부 예측
def rain_OX(weather1_rr, weather2_r_one, model, start=3683, step=90):

    all_predictions = []
        
    for i  in range(start, weatherf_rr.shape[0], step):
        
#강수여부 예측을 위해 학습 시 weather2_r_one의 binary형태의 '예상강수량'은 업데이트가 안 됐으므로 업데이트가 된 weatherf_rr의
#index range 내에서 오류가 발생하지 않도록 모델 학습 시 weather2_r_one의 index에 맞춰줌
        a = weather1_rr.shape[0] - weather2_r_one.shape[0]
        weather1_rrr = weather1_rr.iloc[:weather1_rr.shape[0]-a,:]
        
        train1 = weather1_rrr.iloc[:i, :]
        train2 = weather2_r_one.iloc[:i, :]
        test = weather1_rr.iloc[i:(i+step),:]

#weather1_rr값과 westher2_r_one의 binary값의 상관관계를 학습
        model.fit(train1[weather1_rr.columns], train2['예상강수량'])

#weatherf_rr값을 이용해 예측
        preds = model.predict(test[weather1_rr.columns])
        preds = pd.Series(preds, index = test.index)

        combined = pd.concat([test['평균 상대습도(%)'], preds], axis=1)
        combined.columns = ["actual", '예상강수량']

        all_predictions.append(combined)
    return pd.concat(all_predictions)
#__________________________________________________________________________________________

#업데이트된 weatherf_rr을 이용해 한번에 예측하므로 기온 예측과 달리 index를 늘려주는 과정이 필요없음
rain_OX = rain_OX(weather1_rr, weather2_r_one, lr)
rain_OX = rain_OX.drop('actual', axis=1)

#기온예측 dataframe과 강수여부 dataframe을 합쳐주는 과정에서 강수여부 dataframe은 처음 10년이 비어서 index를 맞춰서 합쳐줌
weatherf_t = weatherf_t.drop(weatherf_t.index[:3683])
weatherff = pd.concat([weatherf_t[weatherf_t.columns], rain_OX[rain_OX.columns]], axis=1)

#합친 dataframe에서 원하는 값의 column들만 모아줌
# weatherfff = weatherff[['예상최저기온', '예상평균기온', '예상최고기온', '예상강수량']].copy()
# weatherfff.columns = ['최저기온', '평균기온', '최고기온', '강수여부']

weatherfff = weatherff[['예상강수량']].copy()
weatherfff.columns = ['강수여부']



#========================================================================================================원하는 날짜의 값을 출력



weatherfff.index.name = 'DATE'

target_date = t_date1+'-'+t_date2+'-'+t_date3
print(weatherfff.query('DATE == @target_date'))