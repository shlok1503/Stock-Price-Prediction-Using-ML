import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import date, timedelta


st.title("Stock Price Prediction")



ticker = st.text_input('Enter Stock Ticker').upper()
option = st.selectbox('Which Model You Want To Use?',('ARIMA', 'LSTM', 'FB Prophet','Others'))

future_dates=[]
for i in range(1,31):
    future_dates.append((date.today()+timedelta(days=i)).isoformat()) 
future_dates=pd.to_datetime(future_dates)

st.write('Prediction for: ',ticker)


try:
    if st.button('Predict'):
        start_date = '1990-01-01'
        end_date = '2023-12-31'
        df = yf.download(ticker, start_date, end_date)
        if option=="ARIMA":
            st.header("Stock Price History")
            fig1 = plt.figure(figsize=(12,10))
            plt.title('Stock Prices History')
            plt.plot(df['Close'])
            plt.xlabel('Date')
            plt.ylabel('Prices ($)')
            st.pyplot(fig1)
            
            step = auto_arima(df['Close'], trace=True, suppress_warnings=True) 
            values=str(step)
            p=int(values[7])
            d=int(values[9])
            q=int(values[11])
            
            
            train = df.iloc[:int(len(df)*0.8)]['Close']
            test =df.iloc[int(len(df)*0.8):]['Close']
            
            
            model = ARIMA(train,order=(p,d,q))
            arima=model.fit()
            
            
            y_hat=pd.DataFrame()
            y_hat['arima']=arima.forecast(len(test),index=test.index)

            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.001)
                my_bar.progress(percent_complete + 1, text=progress_text)
                
            # for Testing data    
            st.header("Actual VS Predicted")
            fig2 = plt.figure(figsize=(12,10))
            plt.plot(np.asarray(test.index),np.asarray(test),color='orange',label="Test")
            plt.plot(np.asarray(y_hat.index),np.asarray(y_hat['arima']),color='green',label="Forecast")
            plt.xlabel('Date')
            plt.ylabel('Prices ($)')
            plt.legend()
            st.pyplot(fig2)
            
            # For Next 30 days
            model = ARIMA(df['Close'],order=(p,d,q))
            arima=model.fit()
            y_hat=pd.DataFrame()
            y_hat['arima']=arima.forecast(30,index=pd.to_datetime(future_dates))
            
            st.write('Prediction for',30,' days is',y_hat['arima'])
            
            # future Prediction Graph
            st.header("Next 30 days Prediction")
            fig3=plt.figure(figsize=(12,10))
            plt.plot(np.asarray(test[-60:].index),np.asarray(test[-60:]),label="Train")
            plt.plot(np.asarray(y_hat.index),np.asarray(y_hat['arima']),color='green',label="Forecast")
            plt.xlabel('Date')
            plt.ylabel('Prices ($)')
            plt.legend()
            st.pyplot(fig3)
            
            
        elif option=="LSTM":
            st.header("Stock Price History")
            fig1 = plt.figure(figsize=(12,10))
            plt.title('Stock Prices History')
            plt.plot(df['Close'])
            plt.xlabel('Date')
            plt.ylabel('Prices ($)')
            st.pyplot(fig1)
            
            
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])
            
            scaler = MinMaxScaler(feature_range=(0,1))
            data_training_array = scaler.fit_transform(data_training)
            
            
            # Testing
            past_60_days = data_training.tail(60)
            final_df=pd.concat([past_60_days,data_testing],ignore_index=True)
            
            input_data= scaler.fit_transform(final_df)
            
            x_test=[]
            y_test=[]
            
            for i in range(60,input_data.shape[0]):
                x_test.append(input_data[i-60:i])
                y_test.append(input_data[i,0])
            
            x_test , y_test = np.array(x_test) , np.array(y_test)
            
            model =load_model("lstm.keras")
            
            y_predicted = model.predict(x_test)
            
            y_predicted=scaler.inverse_transform(y_predicted)
            y_test=y_test.reshape(-1,1)
            y_test=scaler.inverse_transform(y_test)
            
            st.header("Actual VS Predicted")
            fig2= plt.figure(figsize=(12,10))
            plt.plot(y_test , 'r', label='OG price')
            plt.plot(y_predicted , 'b', label='Predicted price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
            
            # Predicting for next 30 Days
            n_steps=60
            pred_days=30
            x_input=input_data[len(input_data)-n_steps:].reshape(1,-1)
            temp_input=list(x_input[0])
            lst_output=[]
            
            i=0
            while(i<pred_days):

                if(len(temp_input)>60):
                    #print(temp_input)
                    x_input=np.array(temp_input[1:])
                    # print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    # print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    # print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    # print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1           

            predictions=scaler.inverse_transform(lst_output)
            st.write('Prediction for',pred_days,' days is',predictions)
            day_new=np.arange(1,61)
            day_pred=np.arange(61,61+pred_days)
            
            st.header("Prediction for Next 30 days")
            fig3= plt.figure(figsize=(12,10))
            plt.plot(np.asarray(df[-60:].index),np.asarray(df.Close[-60:]) , label='OG price')
            plt.plot(np.asarray(future_dates),np.asarray(predictions) , 'green', label='Predicted price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig3)
        
        elif option=='FB Prophet':
            
            st.header("Stock Price History")
            fig1 = plt.figure(figsize=(12,10))
            plt.title('Stock Prices History')
            plt.plot(df['Close'])
            plt.xlabel('Date')
            plt.ylabel('Prices ($)')
            st.pyplot(fig1)
            
            
            df.reset_index(level=0,inplace=True)
            df=df[['Date','Close']]
            df.columns=['ds','y']
            

            
            from prophet import Prophet
            m = Prophet(interval_width=0.95)
            m.fit(df)
            future = m.make_future_dataframe(periods=30,freq='D')
            future.tail(30)
            forecast = m.predict(future)
            y_hat=pd.DataFrame()
            y_hat=forecast[['ds', 'yhat']].tail(30)
            
            st.header("Prediction for Next 30 days")
            fig2= plt.figure(figsize=(12,10))
            plt.plot(np.asarray(df[-60:]['ds']),np.asarray(df[-60:]['y']),label="Actual")
            plt.plot(np.asarray(y_hat['ds']),np.asarray(y_hat['yhat']),label="Forecast")
            plt.legend()
            st.pyplot(fig2)
        elif option=="Others":
            st.info('We are Working on Other Models...', icon="ℹ️")
        else:
            st.warning('Please Choose a Option', icon="⚠️")

            
                
        
    else:
        st.info('Stock Price Prediction', icon="ℹ️")
except:
    st.error('Please Enter The Ticker', icon="🚨")