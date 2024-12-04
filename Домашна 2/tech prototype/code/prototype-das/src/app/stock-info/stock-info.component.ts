import { Component } from '@angular/core';
import {HighchartsChartModule} from "highcharts-angular";
import * as Highcharts from 'highcharts';

@Component({
  selector: 'app-stock-info',
  standalone: true,
  imports: [
    HighchartsChartModule
  ],
  templateUrl: './stock-info.component.html',
  styleUrl: './stock-info.component.scss'
})
export class StockInfoComponent {
  Highcharts: typeof Highcharts = Highcharts;

  chartOptions: Highcharts.Options = {
    title: {
      text: 'Alkaloid Skopje'
    },
    xAxis: {
      type: 'datetime'
    },
    yAxis: {
      title: {
        text: 'Value'
      }
    },
    series: [{
      type: 'line',
      name: 'Value',
      data: [
        [Date.UTC(2023, 11, 4), 18200],
        [Date.UTC(2023, 11, 15), 18400],
        [Date.UTC(2023, 11, 25), 18300],
        [Date.UTC(2024, 0, 10), 18500],
        [Date.UTC(2024, 0, 25), 18700],
        [Date.UTC(2024, 1, 5), 18600],
        [Date.UTC(2024, 1, 15), 19000],
        [Date.UTC(2024, 2, 5), 19200],
        [Date.UTC(2024, 2, 20), 20200],
        [Date.UTC(2024, 3, 1), 19800],
        [Date.UTC(2024, 3, 15), 20000],
        [Date.UTC(2024, 4, 5), 20500],
        [Date.UTC(2024, 4, 20), 21000],
        [Date.UTC(2024, 5, 3), 22000],
        [Date.UTC(2024, 5, 15), 21800],
        [Date.UTC(2024, 6, 1), 22200],
        [Date.UTC(2024, 6, 20), 23000],
        [Date.UTC(2024, 7, 5), 22800],
        [Date.UTC(2024, 7, 25), 23500],
        [Date.UTC(2024, 8, 10), 23800],
        [Date.UTC(2024, 8, 25), 24200],
        [Date.UTC(2024, 9, 5), 24600],
        [Date.UTC(2024, 9, 20), 24500],
        [Date.UTC(2024, 10, 1), 24800],
        [Date.UTC(2024, 10, 15), 25200],
        [Date.UTC(2024, 11, 3), 25500]
      ],
      marker: {
        enabled: true,
        radius: 4
      }
    }]
  };
}
