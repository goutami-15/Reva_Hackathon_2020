from flask import Flask, render_template, request, \
    session, redirect, url_for, flash
from models import plot_daywise_bar, ratio_dayWise, \
    plot_daywise_line, case_over_time_map, top_twenty_hbar, \
    plot_hbar_wm
from deep import worldMap
app = Flask(__name__, static_url_path='/assets',
            static_folder='./covidtrack/assets',
            template_folder='./templates')


country_wise_latest = 'covidtrack/assets/country_wise_latest.csv'
covid_19_clean_complete = 'covidtrack/assets/covid_19_clean_complete.csv'
day_wise = 'covidtrack/assets/day_wise.csv'
full_grouped = 'covidtrack/assets/full_grouped.csv'
usa_county_wise = 'covidtrack/assets/usa_county_wise.csv'
worldometer_data = 'covidtrack/assets/worldometer_data.csv'


@app.route('/')
@app.route('/index.html')
def root():
    bar = plot_daywise_bar(day_wise)
    line = plot_daywise_line(day_wise)

    topTwentyCases = top_twenty_hbar(country_wise_latest, 'Confirmed', 15)
    topTwentyDeaths = top_twenty_hbar(country_wise_latest, 'Deaths', 15)
    topTwentyRecovered = top_twenty_hbar(country_wise_latest, 'Recovered', 15)
    topTwentyActive = top_twenty_hbar(country_wise_latest, 'Active', 15)
    topTwentyNewCases = top_twenty_hbar(country_wise_latest, 'New cases', 15)
    topTwentyNewDeaths = top_twenty_hbar(country_wise_latest, 'New deaths', 15)
    topTwentyDeaths_100_cases = top_twenty_hbar(country_wise_latest, 'Deaths / 100 Cases', 15)
    topTwentyNewRecovered = top_twenty_hbar(country_wise_latest, 'New recovered', 15)
    topTwentyRecovered_100_cases = top_twenty_hbar(country_wise_latest, 'Recovered / 100 Cases', 15)
    topTwenty1_week_change = top_twenty_hbar(country_wise_latest, '1 week change', 15)
    topTwenty1_week_increase = top_twenty_hbar(country_wise_latest, '1 week % increase', 15)

    perMDeaths = plot_hbar_wm(worldometer_data, 'Deaths/1M pop', 15)
    totalTests = plot_hbar_wm(worldometer_data, 'TotalTests', 15)
    perMTests = plot_hbar_wm(worldometer_data, 'Tests/1M pop', 15)

    return render_template('index.html', plot=bar, line=line, topTwentyCases=topTwentyCases, topTwentyDeaths=topTwentyDeaths,
                           topTwentyRecovered=topTwentyRecovered, topTwentyActive=topTwentyActive, topTwentyNewCases=topTwentyNewCases,
                           topTwentyNewDeaths=topTwentyNewDeaths, topTwentyDeaths_100_cases=topTwentyDeaths_100_cases,
                           topTwentyNewRecovered=topTwentyNewRecovered, topTwentyRecovered_100_cases=topTwentyRecovered_100_cases,
                           topTwenty1_week_change=topTwenty1_week_change, topTwenty1_week_increase=topTwenty1_week_increase,

                           perMDeaths=perMDeaths, perMTests=perMTests, totalTests=totalTests
                           )


@app.route('/worldometer')
def worldMap():
    casesMap = case_over_time_map(country_wise_latest, 'Confirmed', 'Confirmed Cases')
    deathsMap = case_over_time_map(country_wise_latest, 'Deaths', 'Deaths')
    deaths_100_Map = case_over_time_map(country_wise_latest, 'Deaths / 100 Cases', 'Deaths / 100 Cases')
    return render_template('worldMap.html',
                           casesMap=casesMap, deathsMap=deathsMap, deaths_100_Map=deaths_100_Map)


if __name__ == '__main__':
    app.run(debug=True)
