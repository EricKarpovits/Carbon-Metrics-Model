"""
Module responsible for plotting the results of the program.
"""

import datetime
from typing import List

import plotly.express as px
import plotly.graph_objects as go

import train
from data import CarbonMetric, DataManager


def line_graph_one_line() -> None:
    """
    Displays a line graph showing the data we will be feeding into the keras model, it is each day
    of 2019 mapped to its total carbon emissions which is the sum of each sector.
    """
    manager = DataManager()
    manager.load_data('carbonmonitor-us_datas_2021-11-04.csv')
    manager.filter(year=2019)
    manager.sum_of_total_emissions('United States')

    total_emissions = manager.metrics
    days = []
    emissions = []

    for day in total_emissions:
        days.append(day.date)
        emissions.append(day.value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=emissions, mode='lines+markers', name='United States'))
    fig.update_layout(title='United States Carbon Emissions in 2019', xaxis_title='Day',
                      yaxis_title='Emission Metric Tonnes')
    fig.show()


def line_graph_predicted_minus_actual(normalized: bool = False) -> None:
    """
    Displays a line graph that shows the difference in the actual carbon emissions and our model's
    predicted emissions for 2020-2021 per day for the sum of all sectors, with the option to use
    normalized or un-normalized predictions.
    """
    if normalized:
        title = 'Difference Predicted vs Actual Carbon Emissions 2020-2021 (Normalized)'
    else:
        title = 'Difference Predicted vs Actual Carbon Emissions 2020-2021'

    actual, predicted = train.main(normalized=normalized)
    index_to_day = index_to_day_list()
    days = []
    emissions = []

    for i in range(len(actual)):
        days.append(index_to_day[i])
        emissions.append(predicted[i] - actual[i])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=emissions,
                             mode='lines+markers', name='Actual'))
    fig.add_shape(type='line', x0=datetime.date(2020, 1, 1), y0=0,
                  x1=datetime.date(2021, 12, 31), y1=0, line=dict(color='Black', ),
                  xref='x', yref='y')
    fig.update_layout(title=title,
                      xaxis_title='Day', yaxis_title='Emission Metric Tonnes')

    fig.show()


def line_graph_predicted_vs_actual(normalized: bool = False) -> None:
    """
    Displays a line graph that shows our predicted values for carbon emissions for 2020-2021 to the
    actual values of emission per day for the sum of all sectors, with the option to use normalized
    or un-normalized predictions.
    """
    if normalized:
        title = 'Predicted vs Actual Carbon Emissions 2020-2021 (Normalized)'
    else:
        title = 'Predicted vs Actual Carbon Emissions 2020-2021'

    actual, predicted = train.main(normalized=normalized)
    index_to_day = index_to_day_list()
    days_actual = []
    emissions_actual = []
    emissions_predicted = []

    for i in range(len(actual)):
        days_actual.append(index_to_day[i])
        emissions_actual.append(actual[i])

    for prediction in predicted:
        emissions_predicted.append(prediction)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_actual, y=emissions_actual,
                             mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=days_actual, y=emissions_predicted,
                             mode='lines+markers', name='Predicted'))
    fig.update_layout(title=title,
                      xaxis_title='Day', yaxis_title='Emission Metric Tonnes')

    fig.show()


def index_to_day_list() -> List[datetime.date]:
    """
    Returns a list of datetime.date values in chronological order.
    The first day of our predicted data starts from February 12, 2020 (index 0).
    """
    manager = DataManager()
    manager.load_data('carbonmonitor-us_datas_2021-11-04.csv')

    start_date = datetime.date(2020, 2, 12)
    delta = datetime.timedelta(days=1)

    new_list = []
    for _ in manager.metrics:
        new_list.append(start_date)
        start_date = start_date + delta

    return new_list


def day_to_index_mapping() -> dict[datetime.date, int]:
    """
    Return a dictionary mapping each datetime.date to an index in chronological order
    (January 1, 2019 is index 0).
    """
    manager = DataManager()
    manager.load_data('carbonmonitor-us_datas_2021-11-04.csv')
    date_mapping = {}
    count = 0

    for metric in manager.metrics:
        if metric.date not in date_mapping:  # Our data is organized by date already so
            # we do not need to worry about the keys (dates) we are adding to the dictionary

            date_mapping[metric.date] = count
            count += 1

    return date_mapping


def slope_secant(day_1: CarbonMetric, day_2: CarbonMetric) -> float:
    """
    Return the rate of change of carbon emissions between a given 2 days (day_1 and day_2)
    using the slope of a line formula.
    """
    date_mapping = day_to_index_mapping()

    x_1 = date_mapping[day_1.date]
    x_2 = date_mapping[day_2.date]
    y_1 = day_1.value
    y_2 = day_2.value
    slope = (y_2 - y_1) / (x_2 - x_1)

    return slope


def state_emissions(year: int) -> dict[str, float]:
    """
    Returns a dictionary mapping carbon emissions for each state

    Preconditions:
      - 2019 <= year <= 2021
    """
    data_manager = DataManager()
    data_manager.load_data('carbonmonitor-us_datas_2021-11-04.csv')
    data = data_manager.metrics

    state_mapping = {}
    for item in data:
        if item.date.year == year:
            if item.state not in state_mapping:
                state_mapping[item.state] = 0

            state_mapping[item.state] += item.value

    return state_mapping


def bar_chart() -> None:
    """
    Displays a barchart showing each state against their total carbon emissions in 2019.
    This is to help understand and represent our data.
    """
    state_mapping = state_emissions(2019)

    state_list = []
    carbon_list = []

    for state in state_mapping:
        if state != 'United States':
            state_list.append(state)
            carbon_list.append(state_mapping[state])

    fig = px.bar(x=state_list, y=carbon_list,
                 labels=dict(x="State", y="CO2 Emissions (Metric Tonnes)"))
    fig.update_layout(title='Carbon Emissions Per State in 2019')
    fig.update_xaxes(type='category')

    fig.show()


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['train', 'data', 'plotly.graph_objects', 'plotly.express', 'datetime'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200', 'C0103']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
