"""
This module is responsible for data reading and handling.
"""

from __future__ import annotations

import copy
import csv
import datetime
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CarbonMetric:
    """
    A custom data type that represents data for a CaronMetrics.

    Instance Attributes:
        - state: The state which the data was recorded
        - date: The date of the metric
        - sector: The sector that emitted the carbon
        - value: Carbon emissions in metric tonnes

    Representation Invariants:
      - self.state != ''
      - self.value >= 0.0
    """
    state: str
    date: datetime.date
    value: float
    sector: Optional[str] = None


class EmptyMetricsError(Exception):
    """Error raised when attempting to get info about an empty list of metrics."""


class DataManager:
    """
    Data Manager responsible for storing carbon metrics and performing computations on them.
    """
    metrics: List[CarbonMetric]

    def __init__(self, metrics: List = None) -> None:
        """ADD A DOCSTRING"""
        if metrics is None:
            metrics = []
        self.metrics = metrics

    def __len__(self) -> int:
        """Returns the length of the metrics"""
        return len(self.metrics)

    def __copy__(self) -> DataManager:
        """Copy the DataManager class and it's metrics"""
        return DataManager(copy.deepcopy(self.metrics))

    def load_data(self, filename: str) -> None:
        """Returns a list of CarbonMetric read from a csv file"""
        # ACCUMULATOR metrics_so_far:
        metrics_so_far = []

        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)

            for row in reader:
                # Loop invariant
                assert len(row) == 5  # each row as 5 columns

                # Sample date: "14/01/2019"
                # Convert into datetime values
                old_date = row[1].split('/')

                day = int(old_date[0])
                month = int(old_date[1])
                year = int(old_date[2])

                date = datetime.date(year, month, day)

                metric = CarbonMetric(
                    state=str(row[0]),
                    date=date,
                    sector=str(row[2]),
                    value=float(row[3]),
                )
                metrics_so_far.append(metric)

        self.metrics = metrics_so_far

    def is_empty(self) -> bool:
        """Returns whether the metrics list is empty."""
        return len(self) == 0

    def copy(self) -> DataManager:
        """Alias to __copy__"""
        return self.__copy__()

    def filter(self, year: Optional[int] = None, month: Optional[int] = None,
               day: Optional[int] = None, state: Optional[str] = None,
               before_day: datetime.date = None, exclude: bool = False) -> None:
        """Filters the data based on the parameters provided."""
        list_so_far = []
        for row in self.metrics:
            if year is not None and (row.date.year == year) == exclude:
                continue
            if month is not None and (row.date.month == month) == exclude:
                continue
            if day is not None and (row.date.day == day) == exclude:
                continue
            if state is not None and (row.state == state) == exclude:
                continue
            if before_day is not None and (row.date <= before_day) == exclude:
                continue
            list_so_far += [row]
        self.metrics = list_so_far

    def get_max_value(self) -> float:
        """Returns the maximum value in carbon metrics"""
        if self.is_empty():
            raise EmptyMetricsError("Cannot get max value of an empty dataset")
        # ACCUMULATOR max_value_so_far:
        max_value_so_far = 0
        for metric in self.metrics:
            if metric.value > max_value_so_far:
                max_value_so_far = metric.value
        return max_value_so_far

    def get_min_value(self) -> float:
        """Returns the minimum value in carbon metrics"""
        if self.is_empty():
            raise EmptyMetricsError("Cannot get min value of an empty dataset")
        # ACCUMULATOR min_value_so_far:
        min_value_so_far = math.inf
        for metric in self.metrics:
            if metric.value < min_value_so_far:
                min_value_so_far = metric.value
        return min_value_so_far

    def normalize_data(self) -> None:
        """Normalizes the data between 0 and 1"""
        maxs = self.get_max_value()
        mins = self.get_min_value()

        for metric in self.metrics:
            metric.value = (metric.value - mins) / (maxs - mins)

    def get_start_date(self) -> datetime.date:
        """Returns the date of the earliest carbon metric"""
        if self.is_empty():
            raise EmptyMetricsError("Cannot get start date of an empty dataset")
        return min([metric.date for metric in self.metrics])

    def get_end_date(self) -> datetime.date:
        """Returns the date of the latest carbon metric"""
        if self.is_empty():
            raise EmptyMetricsError("Cannot get end date of an empty dataset")
        return max([metric.date for metric in self.metrics])

    def sum_of_total_emissions(self, state: str) -> None:
        """
        Returns a list of the daily sums of carbon emissions from all sectors for a state.

        WARNING: implicitly calls self.filter(state=state).
        """
        self.filter(state=state)

        date_to_metric = {}
        for metric in self.metrics:
            if metric.date not in date_to_metric:
                date_to_metric[metric.date] = []
            date_to_metric[metric.date].append(metric)

        start_date = self.get_start_date()
        end_date = self.get_end_date()
        delta = datetime.timedelta(days=1)

        list_so_far = []
        while start_date <= end_date:
            sum_so_far = 0
            for metric in date_to_metric[start_date]:
                sum_so_far += metric.value

            list_so_far.append(CarbonMetric(state=state, date=start_date, value=sum_so_far))
            start_date += delta
        self.metrics = list_so_far


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['__future__', 'csv', 'dataclasses', 'math', 'datetime', 'typing', 'copy'],
        'allowed-io': ['load_data'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200', 'R0913']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
