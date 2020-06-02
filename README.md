# Cellular-Traffic-Prediction-using-LSTM
## Dataset
The data traffic information of 4G LTE networks has been used as
dataset for training and testing. When a subscriber uses mobile data service on their devices,
data will be provided by their nearby 4G cell. The trafiic of any cell within 1 hour is nothing
but the total data capacity of all users served by a cell within an hour. For example, a cell is
serving 50 subscribers, each subscriber in 1-hour uses an average of 10Mb so, the product of
average data consumed by a user (in Megabytes) and the number of users gives the traffic of
that cell. So the traffic of this cell in hours, x = 50 * 10 = 500Mb. Data is collected in
approximately 1 year x 24 hours. The nature of traffic will vary from hour to hour (e.g., 10:00
to 12:00 and 19:00 to 23:00 traffic will be very high, 0:00 to 6:00 traffic will be very low). The
divergence between days of the week (e.g., traffic graphs for office buildings, high traffic on
Monday 14:00 to 18:00 and low on Saturday and Sunday). It differs for special events such as
festivals, holidays, etc. The data consists of 8733 entries from 23-10-2017 to 22-10-2018 (one
year). It is further divided into 2 datasets:
* Training Dataset
* Test Dataset
Test dataset includes the last week of the data (i.e., from 16-10-2018 to 22-10-2018),
and the training data consists of data traffic excluding test data (which is from 23-10-2017 to
15-10-2018). Here we predict the traffic values of 3 LTE cell towers (Cell_000111,
Cell_000112, and Cell_000113) to check the consistency of our LSTM model. The RNN
algorithm requires training and test datasets to predict data traffic. It uses the training dataset to
train the algorithm based on the model and the number of layers required for training to get the
best output.
