import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
 
# Loading the dataset
data = pd.read_csv("DQN_mean_reward-csv.csv")
 
# X axis is price_date
price_date = data['Time step']
 
# Y axis is price closing
price_close = data['Mean Reward']
 
# Plotting the timeseries graph of given dataset
plt.plot(price_date, price_close)
 
# Giving title to the graph
plt.title('Mean episode reward')
 
# rotating the x-axis tick labels at 30degree
# towards right
plt.xticks(rotation=30, ha='right')
 
# Giving x and y label to the graph
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.savefig('mean-reward-dqn.png')