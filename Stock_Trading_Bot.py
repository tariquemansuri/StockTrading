import abc
import threading
import time

import pandas as pd
import numpy as np
import trade as trade
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from AlpacaTradingAPI.rest import REST


class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(key_id='PKZNJMPSKB4MNQ3Q5NW4',
                                                secret_key='fuccHw7r2eLSJwdZbWCZbGR2nb7AFayfUmgGS5S1',
                                                base_url='https://paper-api.alpaca.markets')


class TradingSystem(abc.ABC):
    def __init__(self, api, symbol, time_frame, system_id, system_label):
        # Connect to api
        # Connect to BrokenPipeError
        # Save fields to class
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        # Thread an infinite loop
        thread = threading.Thread(target=self.system_loop)
        thread.start()

    @abc.abstractmethod
    def place_buy_order(self):
        pass

    @abc.abstractmethod
    def place_sell_order(self):
        pass

    @abc.abstractmethod
    def system_loop(self):
        pass


# Class to develop AI portfolio manager
class AIPMDevelopment:
    def __init__(self):
        # Read your data in and split the dependent and independent
        data = pd.read_csv('IBM.csv')
        x = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        # Train test spit
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        # Create the sequential
        network = Sequential()

        # Create the structure of the neural network
        network.add(Dense(1, input_shape=(1,), activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(1, activation='tanh'))

        # Compile the model
        network.compile(
            optimizer='rmsprop',
            loss='hinge',
            metrics=['accuracy']
        )

        # Train the model
        network.fit(x_train.values, y_train.values, epochs=100)

        # Evaluate the predictions of the model
        y_pred = network.predict(x_test.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y_test, y_pred))

        # Save structure to json
        model = network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model)

        # Save weights to HDF5
        network.save_weights("weigths.h5")


# AI Portfolio Manager
class PortfolioManagementModel:
    def __init__(self):
        # Data in to test that the saving of weights worked
        data = pd.read_csv('IBM.csv')
        x = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        # Read structure from json
        json_file = open('model.json', 'r')
        json = json_file.read()
        json_file.close()
        self.network = model_from_json(json)

        # Read weights from HDF5
        self.network.load_weights("weights.h5")

        # Verify weights and strcuture are loaded
        y_pred = self.network.predict(x.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y, y_pred))


class PortfolioManagementSystem(TradingSystem):
    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 86400, 1, 'AI_PM')
        self.AI = PortfolioManagementModel()

    def place_buy_order(self):
        self.api.submit_order(symbol='IBM', qty=1, side='buy', type='market', time_in_force='day')

    def place_sell_order(self):
        self.api.submit_order(symbol='IBM', qty=1, side='sell', type='market', time_in_force='day')

    def system_loop(self):
        # Variables for weekly close
        this_weeks_close = 0
        last_weeeks_close = 0
        delta = 0
        day_count = 0

        while True:
            # Wait a day to request more data
            time.sleep(1440)
            # Request EoD data for IBM
            data_req = self.api.get_barset('IBM', timeframe='1D', limit=1).df
            # Construct dataframe to predict
            x = pd.DataFrame(data=[[data_req['IBM']['close'][0]]], columns='Close'.split())
            if day_count == 7:
                day_count = 0
                last_weeeks_close = this_weeks_close
                this_weeks_close = x['Close']
                delta = this_weeks_close - last_weeeks_close

                # AI choosing to buy, sell, or hold
                if np.around(self.AI.network.predict([delta])) <= -.5:
                    self.place_sell_order()
                elif np.around(self.AI.network.predict([delta]) >= .5):
                    self.place_buy_order()


def mainFunc():
    AIPMDevelopment()
    PortfolioManagementModel()
    PortfolioManagementSystem()
