#!/usr/bin/env python
# coding: utf-8

# In[123]:


#백테스팅 결과 참고 후, 수정 필요
#유니버스 - 소형주
#세 가지 팩터만 활용하기로 함

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock
import plotly.offline as pyo # jupyter notebook 에서 보여지도록 설정하는 부분 (가끔 안나올 때, 이 명령을 하면 됨)
pyo.init_notebook_mode()

def perform_screening(end_date):
    end_date_datetime = datetime.strptime(end_date, "%Y%m%d")
    start_date_datetime = end_date_datetime - timedelta(days=30)
    start_date = start_date_datetime.strftime("%Y%m%d")

    # 시가총액 데이터 가져오기
    
    kospi_small = stock.get_index_portfolio_deposit_file("1004")

    market_cap_data = stock.get_market_cap(end_date, market="ALL")
    market_cap_data = market_cap_data[['종가', '시가총액']]

    filtered_market_cap_data = market_cap_data.loc[kospi_small]
    
    gains = []
    filtered_tickers = filtered_market_cap_data.index

    for ticker in filtered_tickers:
        df = stock.get_market_ohlcv(start_date, end_date, ticker, "m")
        gain = ((df.iloc[1,3] - df.iloc[0,3]) / df.iloc[0,3]) * 100
        gains.append(gain)

    result_df = pd.DataFrame({'ticker': filtered_tickers, 'gain': gains})
    result_df.set_index('ticker', inplace=True)
    gain_25 = result_df['gain'].quantile(0.25)
    selected_rows = result_df['gain'] < gain_25
    temp_df = result_df[selected_rows]
    selected_tickers = temp_df.index.tolist()

    # 평균값 계산
    revisions = []

    for ticker in selected_tickers:
        data = stock.get_market_ohlcv(start_date, end_date, ticker, "d")
        mean_price = data['종가'].mean()
        close_temp = stock.get_market_ohlcv(start_date, end_date, ticker, "d")
        close = close_temp.iloc[0,3]
        revision = (close / mean_price) * 100
        revisions.append(revision)

    result_df = pd.DataFrame({'ticker': selected_tickers, 'revision': revisions})
    result_df.set_index('ticker', inplace=True)
    revision_25 = result_df['revision'].quantile(0.25)
    selected_rows = result_df['revision'] < revision_25
    revision_df = result_df[selected_rows]

    # 평균값 계산
    mean_values = []

    for ticker in revision_df.index:
        data1 = stock.get_market_trading_value_by_date(start_date, end_date, ticker)
        data2 = stock.get_market_cap(start_date, end_date, ticker)['시가총액']
        data1['개인순매수'] = (data1['개인'] / data2) * 100
        mean = data1['개인순매수'].mean()
        mean_values.append(mean)

    final_df = pd.DataFrame({'ticker': revision_df.index, 'mean': mean_values})
    final_df.set_index('ticker', inplace=True)
    final_75 = final_df['mean'].quantile(0.75)
    selected_rows = final_df['mean'] > final_75
    screening = final_df[selected_rows]

    return screening

# 스크리닝 함수 호출
end_date = "20240322"
screened_stocks = perform_screening(end_date)


# In[124]:


print(screened_stocks.index)


# In[150]:


tickers = screened_stocks.index
ticker_names = pd.Series(index=tickers)

for ticker in tickers:
    ticker_names[ticker] = stock.get_market_ticker_name(ticker)

ticker_names.to_list()


# In[154]:


ticker_names = pd.Series(ticker_names,
                         index=screened_stocks.index)
print(ticker_names)


# ### 백테스트 코드

# In[138]:


# 인덱스들 종가 불러오기
final_values = []
for ticker in screened_stocks.index:
    data = stock.get_market_ohlcv("20240325", "20240325", ticker, "d")
    final = data.iloc[0,3]
    final_values.append(final)

final_df = pd.DataFrame({'ticker': screened_stocks.index, '종가': final_values})
cash = 100000000 # 모의투자 시작금액인 1억으로 시작하겠습니다.
money = cash

if len(screened_stocks.index)==0:
    allocation=0
else:
    allocation = money / len(screened_stocks.index) # 동일 비중이기 때문에 보유 현금 / 투자할 종목수 로 나눠줍니다.

final_df['매수 수량'] = allocation / final_df['종가']
final_df


# In[157]:


final_df['기업명'] = ticker_names.loc[final_df['ticker']].values
final_df


# In[127]:


import FinanceDataReader as fdr #벤치마킹 지수

ref = fdr.DataReader(
    symbol = 'KS11',
    start = '20100319',
    end = '20240325')

print(ref)


# In[128]:


import pandas as pd
import numpy as np

# 종가를 기준으로 수익률, 누적 수익률, 로그 수익률, 누적 로그 수익률을 계산하는 함수 정의
def calculate_returns(df):
    # 수익률 계산
    returns = df['Close'].pct_change() * 100  # 변화율을 백분율로 표현
    cum_returns = (1 + returns / 100).cumprod() - 1  # 누적 수익률 계산
    
    # 로그 수익률 계산
    log_returns = np.log(1 + returns / 100)
    cum_log_returns = log_returns.cumsum()  # 누적 로그 수익률 계산
    
    # 데이터프레임으로 결과 반환
    result_df = pd.DataFrame({
        '수익률': returns,
        '누적 수익률': cum_returns,
        '로그 수익률': log_returns,
        '누적 로그 수익률': cum_log_returns
    }, index=df.index)
    
    return result_df

# 주어진 데이터프레임에 대해 수익률 계산
ref_returns = calculate_returns(ref)

ref_returns


# In[129]:


stock_codes = screened_stocks.index

result_data = []

for code in stock_codes:
    data = stock.get_market_ohlcv("20100319", "20240325", code, "m")
    
    # 데이터가 존재하는 경우에만 계산하고 결과 리스트에 추가합니다.
    if not data.empty:
        # 종가와 시가의 차이를 계산하여 새로운 값을 구합니다.
        new_value = data['종가']
        
        # 결과 리스트에 종목 코드를 인덱스로 하고, 날짜를 칼럼으로 하는 데이터프레임을 추가합니다.
        result_data.append(pd.DataFrame(new_value.values, index=data.index, columns=[code]))

final_result = pd.concat(result_data, axis=1)
print(final_result)


# In[130]:


import pandas as pd
import numpy as np

stock_codes = screened_stocks.index

result_data = []

for code in stock_codes:
    data = stock.get_market_ohlcv("20100319", "20240325", code, "m")
    
    # 데이터가 존재하는 경우에만 계산하고 결과 리스트에 추가합니다.
    if not data.empty:
        # 종가와 시가의 차이를 계산하여 새로운 값을 구합니다.
        new_value = data['종가']
        
        # 수익률 계산
        returns = new_value.pct_change() * 100  # 변화율을 백분율로 표현
        cum_returns = (1 + returns / 100).cumprod() - 1  # 누적 수익률 계산
        
        # 로그 수익률 계산
        log_returns = np.log(1 + returns / 100)
        cum_log_returns = log_returns.cumsum()  # 누적 로그 수익률 계산
        
        # 결과 리스트에 종목 코드를 인덱스로 하고, 날짜를 칼럼으로 하는 데이터프레임을 추가합니다.
        result_df = pd.DataFrame(new_value.values, index=data.index, columns=[code])
        result_df['수익률'] = returns  # 수익률 칼럼 추가
        result_df['누적 수익률'] = cum_returns  # 누적 수익률 칼럼 추가
        result_df['로그 수익률'] = log_returns  # 로그 수익률 칼럼 추가
        result_df['누적 로그 수익률'] = cum_log_returns  # 누적 로그 수익률 칼럼 추가
        result_data.append(result_df)
        
final_result = pd.concat(result_data, axis=1)
final_result


# In[131]:


# 각 칼럼들의 평균을 계산하여 새로운 데이터프레임으로 만듭니다.
mean_result = final_result.groupby(level=0, axis=1).mean()
mean_result = mean_result[['누적 로그 수익률', '누적 수익률', '로그 수익률', '수익률']]
mean_result


# In[132]:


import plotly.graph_objs as go

# mean_result 데이터프레임으로부터 날짜와 각 열 데이터 추출
dates = mean_result.index
cumulative_log_returns = mean_result['누적 로그 수익률']
cumulative_returns = mean_result['누적 수익률']
cumulative_log_returns1 = ref_returns['누적 로그 수익률']
cumulative_returns1 = ref_returns['누적 수익률']
# 라인 그래프 생성
trace1 = go.Scatter(x=dates, y=cumulative_log_returns, mode='lines', name='포트폴리오 누적 로그 수익률')
trace2 = go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='포트폴리오 누적 수익률')
trace3 = go.Scatter(x=dates, y=cumulative_log_returns1, mode='lines', name='코스피 누적 로그 수익률')
trace4 = go.Scatter(x=dates, y=cumulative_returns1, mode='lines', name='코스피 누적 수익률')

# 그래프 레이아웃 설정
layout = go.Layout(title='<b> 누적 및 로그 수익률 <b>', xaxis=dict(title='날짜'), yaxis=dict(title='수익률'))

# 그래프 생성 및 표시
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()


# In[133]:


import plotly.graph_objs as go
import plotly.io as pio

# Set Seaborn template
pio.templates["seaborn"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif"),
        title=dict(font=dict(size=20)),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16)),
        legend=dict(font=dict(size=14)),
    )
)
pio.templates.default = "seaborn"

# mean_result 데이터프레임으로부터 날짜와 각 열 데이터 추출
dates = mean_result.index
cumulative_log_returns = mean_result['누적 로그 수익률']
cumulative_returns = mean_result['누적 수익률']
cumulative_log_returns1 = ref_returns['누적 로그 수익률']
cumulative_returns1 = ref_returns['누적 수익률']

# 라인 그래프 생성
trace1 = go.Scatter(x=dates, y=cumulative_log_returns, mode='lines', name='포트폴리오 누적 로그 수익률')
trace2 = go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='포트폴리오 누적 수익률')
trace3 = go.Scatter(x=dates, y=cumulative_log_returns1, mode='lines', name='코스피 누적 로그 수익률')
trace4 = go.Scatter(x=dates, y=cumulative_returns1, mode='lines', name='코스피 누적 수익률')

# 그래프 레이아웃 설정
layout = go.Layout(
    title='<b>누적 및 로그 수익률</b>',
    xaxis=dict(title='날짜', showgrid=True, tickfont=dict(size=12)),
    yaxis=dict(title='수익률', showgrid=True, tickfont=dict(size=12)),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'),
    plot_bgcolor='rgba(0, 0, 0, 0)'
)

# 그래프 생성 및 표시
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()


# In[134]:


#백테스트 전체 코드 
import FinanceDataReader as fdr #벤치마킹 지수

ref = fdr.DataReader(
    symbol = 'KS11',
    start = '20100319',
    end = '20240325')

import pandas as pd
import numpy as np

# 종가를 기준으로 수익률, 누적 수익률, 로그 수익률, 누적 로그 수익률을 계산하는 함수 정의
def calculate_returns(df):
    # 수익률 계산
    returns = df['Close'].pct_change() * 100  # 변화율을 백분율로 표현
    cum_returns = (1 + returns / 100).cumprod() - 1  # 누적 수익률 계산
    
    # 로그 수익률 계산
    log_returns = np.log(1 + returns / 100)
    cum_log_returns = log_returns.cumsum()  # 누적 로그 수익률 계산
    
    # 데이터프레임으로 결과 반환
    result_df = pd.DataFrame({
        '수익률': returns,
        '누적 수익률': cum_returns,
        '로그 수익률': log_returns,
        '누적 로그 수익률': cum_log_returns
    }, index=df.index)
    
    return result_df

# 주어진 데이터프레임에 대해 수익률 계산
ref_returns = calculate_returns(ref)

#포트폴리오 계산
stock_codes = screened_stocks.index

result_data = []

for code in stock_codes:
    data = stock.get_market_ohlcv("20100319", "20240325", code, "m")
    
    # 데이터가 존재하는 경우에만 계산하고 결과 리스트에 추가합니다.
    if not data.empty:
        # 종가와 시가의 차이를 계산하여 새로운 값을 구합니다.
        new_value = data['종가']
        
        # 결과 리스트에 종목 코드를 인덱스로 하고, 날짜를 칼럼으로 하는 데이터프레임을 추가합니다.
        result_data.append(pd.DataFrame(new_value.values, index=data.index, columns=[code]))

final_result = pd.concat(result_data, axis=1)

stock_codes = screened_stocks.index

result_data = []

for code in stock_codes:
    data = stock.get_market_ohlcv("20100319", "20240325", code, "m")
    
    # 데이터가 존재하는 경우에만 계산하고 결과 리스트에 추가합니다.
    if not data.empty:
        # 종가와 시가의 차이를 계산하여 새로운 값을 구합니다.
        new_value = data['종가']
        
        # 수익률 계산
        returns = new_value.pct_change() * 100  # 변화율을 백분율로 표현
        cum_returns = (1 + returns / 100).cumprod() - 1  # 누적 수익률 계산
        
        # 로그 수익률 계산
        log_returns = np.log(1 + returns / 100)
        cum_log_returns = log_returns.cumsum()  # 누적 로그 수익률 계산
        
        # 결과 리스트에 종목 코드를 인덱스로 하고, 날짜를 칼럼으로 하는 데이터프레임을 추가합니다.
        result_df = pd.DataFrame(new_value.values, index=data.index, columns=[code])
        result_df['수익률'] = returns  # 수익률 칼럼 추가
        result_df['누적 수익률'] = cum_returns  # 누적 수익률 칼럼 추가
        result_df['로그 수익률'] = log_returns  # 로그 수익률 칼럼 추가
        result_df['누적 로그 수익률'] = cum_log_returns  # 누적 로그 수익률 칼럼 추가
        result_data.append(result_df)
        
final_result = pd.concat(result_data, axis=1)
final_result

mean_result = final_result.groupby(level=0, axis=1).mean()
mean_result = mean_result[['누적 로그 수익률', '누적 수익률', '로그 수익률', '수익률']]


# In[135]:


import plotly.graph_objs as go
import plotly.io as pio

# Set Seaborn template
pio.templates["seaborn"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif"),
        title=dict(font=dict(size=20)),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16)),
        legend=dict(font=dict(size=12)),
    )
)
pio.templates.default = "seaborn"

# mean_result 데이터프레임으로부터 날짜와 각 열 데이터 추출
dates = mean_result.index
cumulative_log_returns = mean_result['누적 로그 수익률']
cumulative_returns = mean_result['누적 수익률']
cumulative_log_returns1 = ref_returns['누적 로그 수익률']
cumulative_returns1 = ref_returns['누적 수익률']

# 라인 그래프 생성
trace1 = go.Scatter(x=dates, y=cumulative_log_returns, mode='lines', name='포트폴리오 누적 로그 수익률')
trace2 = go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='포트폴리오 누적 수익률')
trace3 = go.Scatter(x=dates, y=cumulative_log_returns1, mode='lines', name='코스피 누적 로그 수익률')
trace4 = go.Scatter(x=dates, y=cumulative_returns1, mode='lines', name='코스피 누적 수익률')

# 그래프 레이아웃 설정
layout = go.Layout(
    title='<b>누적 및 로그 수익률</b>',
    xaxis=dict(title='날짜', tickmode='auto', dtick='M12', tickformat='%Y', showgrid=True, tickfont=dict(size=12)),
    yaxis=dict(title='수익률', showgrid=True, tickfont=dict(size=12)),
    legend=dict(x=1.02, y=1, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'),
    plot_bgcolor='rgba(0, 0, 0, 0)'
)

# 그래프 생성 및 표시
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()


# In[136]:


import plotly.graph_objs as go
import plotly.io as pio

# Set Seaborn template
pio.templates["seaborn"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif"),
        title=dict(font=dict(size=20)),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16)),
        legend=dict(font=dict(size=12)),
    )
)
pio.templates.default = "seaborn"

# mean_result 데이터프레임으로부터 날짜와 각 열 데이터 추출
dates = mean_result.index
cumulative_log_returns = mean_result['누적 로그 수익률']
cumulative_returns = mean_result['누적 수익률']
cumulative_log_returns1 = ref_returns['누적 로그 수익률']
cumulative_returns1 = ref_returns['누적 수익률']

# 라인 그래프 생성
trace1 = go.Scatter(x=dates, y=cumulative_log_returns, mode='lines', name='포트폴리오 누적 로그 수익률')
trace2 = go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='포트폴리오 누적 수익률')
trace3 = go.Scatter(x=dates, y=cumulative_log_returns1, mode='lines', name='코스피 누적 로그 수익률')
trace4 = go.Scatter(x=dates, y=cumulative_returns1, mode='lines', name='코스피 누적 수익률')

# 그래프 레이아웃 설정
layout = go.Layout(
    title='<b>누적 및 로그 수익률</b>',
    xaxis=dict(title='날짜', tickmode='auto', dtick='M12', tickformat='%Y', showgrid=True, tickfont=dict(size=12)),
    yaxis=dict(title='수익률', showgrid=True, tickfont=dict(size=12)),
    legend=dict(x=1.02, y=1, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'),
    plot_bgcolor='rgba(0, 0, 0, 0)'
)

# 그래프 생성 및 표시
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()

