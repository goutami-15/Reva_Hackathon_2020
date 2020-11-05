import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from markupsafe import Markup
from matplotlib import ticker
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta, date
from scipy.interpolate import make_interp_spline, BSpline
import json
import requests
import calmap

# import tensorflow as tf
# from tf.keras.layers import Input, Dense, Activation, LeakyReLu, Dropout
# from tf.keras import models
# from tf.keras.optimizers import RMSprop, Adam


# Retriving Dataset
df_confirmed = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# Depricated
df_recovered = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",
                       parse_dates=['Last_Update'])

# new dataset
df_covid19 = df_covid19.drop(["People_Tested", "People_Hospitalized", "UID", "ISO3", "Mortality_Rate"], axis=1)

# preprocessing
df_confirmed = df_confirmed.rename(columns={"Province/State": "state", "Country/Region": "country"})
df_deaths = df_deaths.rename(columns={"Province/State": "state", "Country/Region": "country"})
df_covid19 = df_covid19.rename(columns={"Country_Region": "country"})
df_covid19["Active"] = df_covid19["Confirmed"] - df_covid19["Recovered"] - df_covid19["Deaths"]

# Changing the conuntry names as required by pycountry_convert Lib
df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"
df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"
df_covid19.loc[df_covid19['country'] == "US", "country"] = "USA"
df_table.loc[df_table['Country_Region'] == "US", "Country_Region"] = "USA"
# df_recovered.loc[df_recovered['country'] == "US", "country"] = "USA"


df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'
df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'
df_covid19.loc[df_covid19['country'] == "Korea, South", "country"] = "South Korea"
df_table.loc[df_table['Country_Region'] == "Korea, South", "Country_Region"] = "South Korea"
# df_recovered.loc[df_recovered['country'] == 'Korea, South', "country"] = 'South Korea'

df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_covid19.loc[df_covid19['country'] == "Taiwan*", "country"] = "Taiwan"
df_table.loc[df_table['Country_Region'] == "Taiwan*", "Country_Region"] = "Taiwan"
# df_recovered.loc[df_recovered['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"
df_table.loc[df_table['Country_Region'] == "Congo (Kinshasa)", "Country_Region"] = "Democratic Republic of the Congo"
# df_recovered.loc[df_recovered['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_covid19.loc[df_covid19['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_table.loc[df_table['Country_Region'] == "Cote d'Ivoire", "Country_Region"] = "Côte d'Ivoire"
# df_recovered.loc[df_recovered['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"
df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"
df_covid19.loc[df_covid19['country'] == "Reunion", "country"] = "Réunion"
df_table.loc[df_table['Country_Region'] == "Reunion", "Country_Region"] = "Réunion"
# df_recovered.loc[df_recovered['country'] == "Reunion", "country"] = "Réunion"

df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"
df_table.loc[df_table['Country_Region'] == "Congo (Brazzaville)", "Country_Region"] = "Republic of the Congo"
# df_recovered.loc[df_recovered['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_covid19.loc[df_covid19['country'] == "Bahamas, The", "country"] = "Bahamas"
df_table.loc[df_table['Country_Region'] == "Bahamas, The", "Country_Region"] = "Bahamas"
# df_recovered.loc[df_recovered['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'
df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'
df_covid19.loc[df_covid19['country'] == "Gambia, The", "country"] = "Gambia"
df_table.loc[df_table['Country_Region'] == "Gambia", "Country_Region"] = "Gambia"
# df_recovered.loc[df_recovered['country'] == 'Gambia, The', "country"] = 'Gambia'

# getting all countries
countries = np.asarray(df_confirmed["country"])
countries1 = np.asarray(df_covid19["country"])
# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU': 'Europe',
    'na': 'Others'
}


# Defininng Function for getting continent code for country.
def country_to_continent_code(country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
    except:
        return 'na'


# Collecting Continent Information
df_confirmed.insert(2, "continent", [continents[country_to_continent_code(country)] for country in countries[:]])
df_deaths.insert(2, "continent", [continents[country_to_continent_code(country)] for country in countries[:]])
df_covid19.insert(1, "continent", [continents[country_to_continent_code(country)] for country in countries1[:]])
df_table.insert(1, "continent",
                [continents[country_to_continent_code(country)] for country in df_table["Country_Region"].values])
# df_recovered.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]] )

df_table = df_table[df_table["continent"] != "Others"]
df_deaths[df_deaths["continent"] == 'Others']
df_confirmed = df_confirmed.replace(np.nan, '', regex=True)
df_deaths = df_deaths.replace(np.nan, '', regex=True)


def plot_params(ax, axis_label=None, plt_title=None, label_size=15, axis_fsize=15, title_fsize=20, scale='linear'):
    # Tick-Parameters
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', width=1, labelsize=label_size)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3, color='0.8')

    # Grid
    plt.grid(lw=1, ls='-', c="0.7", which='major')
    plt.grid(lw=1, ls='-', c="0.9", which='minor')

    # Plot Title
    plt.title(plt_title, {'fontsize': title_fsize})

    # Yaxis sacle
    plt.yscale(scale)
    plt.minorticks_on()
    # Plot Axes Labels
    xl = plt.xlabel(axis_label[0], fontsize=axis_fsize)
    yl = plt.ylabel(axis_label[1], fontsize=axis_fsize)


def visualize_covid_cases(confirmed, deaths, continent=None, country=None, state=None, period=None, figure=None,
                          scale="linear"):
    x = 0
    if figure == None:
        f = plt.figure(figsize=(10, 10))
        # Sub plot
        ax = f.add_subplot(111)
    else:
        f = figure[0]
        # Sub plot
        ax = f.add_subplot(figure[1], figure[2], figure[3])
    ax.set_axisbelow(True)
    plt.tight_layout(pad=10, w_pad=5, h_pad=5)

    stats = [confirmed, deaths]
    label = ["Confirmed", "Deaths"]

    if continent != None:
        params = ["continent", continent]
    elif country != None:
        params = ["country", country]
    else:
        params = ["All", "All"]
    color = ["darkcyan", "crimson"]
    marker_style = dict(linewidth=3, linestyle='-', marker='o', markersize=4, markerfacecolor='#ffffff')
    for i, stat in enumerate(stats):
        if params[1] == "All":
            cases = np.sum(np.asarray(stat.iloc[:, 5:]), axis=0)[x:]
        else:
            cases = np.sum(np.asarray(stat[stat[params[0]] == params[1]].iloc[:, 5:]), axis=0)[x:]
        date = np.arange(1, cases.shape[0] + 1)[x:]
        plt.plot(date, cases, label=label[i] + " (Total : " + str(cases[-1]) + ")", color=color[i], **marker_style)
        plt.fill_between(date, cases, color=color[i], alpha=0.3)

    if params[1] == "All":
        Total_confirmed = np.sum(np.asarray(stats[0].iloc[:, 5:]), axis=0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1].iloc[:, 5:]), axis=0)[x:]
    else:
        Total_confirmed = np.sum(np.asarray(stats[0][stat[params[0]] == params[1]].iloc[:, 5:]), axis=0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1][stat[params[0]] == params[1]].iloc[:, 5:]), axis=0)[x:]

    text = "From " + stats[0].columns[5] + " to " + stats[0].columns[-1] + "\n"
    text += "Mortality rate : " + str(int(Total_deaths[-1] / (Total_confirmed[-1]) * 10000) / 100) + "\n"
    text += "Last 5 Days:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-6]) + "\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-6]) + "\n"
    text += "Last 24 Hours:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-2]) + "\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-2]) + "\n"

    plt.text(0.02, 0.78, text, fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.4))

    # Plot Axes Labels
    axis_label = ["Days (" + df_confirmed.columns[5] + " - " + df_confirmed.columns[-1] + ")", "No of Cases"]

    # Plot Parameters
    plot_params(ax, axis_label, scale=scale)

    # Plot Title
    if params[1] == "All":
        plt.title("COVID-19 Cases World", {'fontsize': 25})
    else:
        plt.title("COVID-19: " + params[1], {'fontsize': 25})

    # Legend Location
    l = plt.legend(loc="best", fontsize=15)

    if figure == None:
        plt.show()


def get_total_cases(cases, country="All"):
    if (country == "All"):
        return np.sum(np.asarray(cases.iloc[:, 5:]), axis=0)[-1]
    else:
        return np.sum(np.asarray(cases[cases["country"] == country].iloc[:, 5:]), axis=0)[-1]


def get_mortality_rate(confirmed, deaths, continent=None, country=None):
    if continent != None:
        params = ["continent", continent]
    elif country != None:
        params = ["country", country]
    else:
        params = ["All", "All"]

    if params[1] == "All":
        Total_confirmed = np.sum(np.asarray(confirmed.iloc[:, 5:]), axis=0)
        Total_deaths = np.sum(np.asarray(deaths.iloc[:, 5:]), axis=0)
        mortality_rate = np.round((Total_deaths / (Total_confirmed + 1.01)) * 100, 2)
    else:
        Total_confirmed = np.sum(np.asarray(confirmed[confirmed[params[0]] == params[1]].iloc[:, 5:]), axis=0)
        Total_deaths = np.sum(np.asarray(deaths[deaths[params[0]] == params[1]].iloc[:, 5:]), axis=0)
        mortality_rate = np.round((Total_deaths / (Total_confirmed + 1.01)) * 100, 2)

    return np.nan_to_num(mortality_rate)


def dd(date1, date2):
    return (datetime.strptime(date1, '%m/%d/%y') - datetime.strptime(date2, '%m/%d/%y')).days


out = ""  # +"output/"

df_countries_cases = df_covid19.copy().drop(['Lat', 'Long_', 'continent', 'Last_Update'], axis=1)
df_countries_cases.index = df_countries_cases["country"]
df_countries_cases = df_countries_cases.drop(['country'], axis=1)

df_continents_cases = df_covid19.copy().drop(['Lat', 'Long_', 'country', 'Last_Update'], axis=1)
df_continents_cases = df_continents_cases.groupby(["continent"]).sum()

df_countries_cases.fillna(0, inplace=True)
df_continents_cases.fillna(0, inplace=True)

df_t = pd.DataFrame(pd.to_numeric(df_countries_cases.sum()), dtype=np.float64).transpose()
df_t["Mortality Rate (per 100)"] = np.round(100 * df_t["Deaths"] / df_t["Confirmed"], 2)


def worldMap():
    world_map = folium.Map(location=[10, 0], tiles="cartodbpositron", zoom_start=2, max_zoom=6, min_zoom=2)
    for i in range(0, len(df_confirmed)):
        folium.Circle(
            location=[df_confirmed.iloc[i]['Lat'], df_confirmed.iloc[i]['Long']],
            tooltip="<h5 style='text-align:center;font-weight: bold'>" + df_confirmed.iloc[i]['country'] + "</h5>" +
                    "<div style='text-align:center;'>" + str(np.nan_to_num(df_confirmed.iloc[i]['state'])) + "</div>" +
                    "<hr style='margin:10px;'>" +
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>" +
                    "<li>Confirmed: " + str(df_confirmed.iloc[i, -1]) + "</li>" +
                    "<li>Deaths:   " + str(df_deaths.iloc[i, -1]) + "</li>" +
                    "<li>Mortality Rate:   " + str(
                np.round(df_deaths.iloc[i, -1] / (df_confirmed.iloc[i, -1] + 1.00001) * 100, 2)) + "</li>" +
                    "</ul>"
            ,
            radius=(int((np.log(df_confirmed.iloc[i, -1] + 1.00001))) + 0.2) * 50000,
            color='#ff6600',
            fill_color='#ff8533',
            fill=True).add_to(world_map)
        # first, force map to render as HTML, for us to dissect
        _ = world_map._repr_html_()

        # get definition of map in body
        map_div = Markup(world_map.get_root().html.render())

        # html to be included in header
        hdr_txt = Markup(world_map.get_root().header.render())

        # html to be included in <script>
        script_txt = Markup(world_map.get_root().script.render())

    return [map_div, script_txt]
