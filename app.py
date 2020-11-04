## Before starting the code make sure to install flask and folium
pip install flask
pip install folium
## Load the dataset and collect the top 15 regions having the largest corona cases
import pandas as pd
corona_df = pd.read_csv('dataset-1.csv') ## You can use any other datasets as well
by_country = corona_df.groupby('Country_Region').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]
cdf = by_country.nlargest(n, 'Confirmed')[['Confirmed']]
## create a function that will return the updated data frame, cdf
def find_top_confirmed(n = 15):
  import pandas as pd
  corona_df = pd.read_csv('dataset.csv')
  by_country = corona_df.groupby('Country_Region').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]
  cdf = by_country.nlargest(n, 'Confirmed')[['Confirmed']]
  return cdf
