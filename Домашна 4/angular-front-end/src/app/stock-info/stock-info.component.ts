import {Component} from '@angular/core';
import {HighchartsChartModule} from "highcharts-angular";
import * as Highcharts from 'highcharts';
import {SpeedometerComponent} from "./speedometer/speedometer.component";
import {ActivatedRoute, Router} from "@angular/router";
import {HttpClient} from "@angular/common/http";
import {NgClass, NgIf} from "@angular/common";
import {timeout} from "rxjs";


@Component({
    selector: 'app-stock-info',
    standalone: true,
    imports: [
        HighchartsChartModule,
        SpeedometerComponent,
        NgIf,
        NgClass
    ],
    templateUrl: './stock-info.component.html',
    styleUrl: './stock-info.component.scss'
})
export class StockInfoComponent {
    isLoading = true;

    currentSpeed = 20;
    Highcharts: typeof Highcharts = Highcharts;
    chartOptions: Highcharts.Options = {};

    //////////////////////////

    timeFrame: string = '1D';
    refresh : boolean = false;

    //////////////////////////

    symbol: string | null = null; // Holds the query parameter
    stockData: any = null; // Holds the stock data

    //////////////////////////

    companyName: string = '';
    currentPrice: string = '';
    allTimeHigh: string = '';
    allTimeLow: string = '';
    revenueGenerated: string = '';
    averagePrice: string = '';

    //////////////////////////

    rsiValue: string = '';
    macdValue: string = '';
    stochValue: string = '';
    williamsValue: string = '';
    rateOfChangeValue: string = '';

    sma10: number = 0;
    sma20: number = 0;
    sma50: number = 0;
    ema10: number = 0;
    ema20: number = 0;


    oscillatorSIGNAL: string = '';
    maSIGNAL: string = '';

    //////////////////////////


    pricePredictionFromModel: string = '';
    buyOrSellPredictionFromModel: string = '';

    //////////////////////////

    constructor(private route: ActivatedRoute, private http: HttpClient, private router: Router) {
    }

    async ngOnInit(): Promise<void> {
        // Subscribe to query parameters
        this.route.queryParams.subscribe(async (params) => {
            this.symbol = params['symbol']; // 'symbol' is the query parameter key

            if (this.symbol) {
                this.isLoading = true; // Ensure the loading spinner is shown
                try {
                    await this.loadStockData(this.symbol); // Wait for API call to complete
                    this.isLoading = false; // Hide the spinner after loading is done
                    // Hide the spinner after loading is done
                } catch (error) {
                    console.error('Failed to load stock data:', error);
                    this.isLoading = false; // Hide the spinner even if an error occurs
                }
            }
        });
    }

    loadStockData(symbol: string): Promise<void> {
        const apiUrl = `https://stefan155-das-homework-api.hf.space/stock-data/${symbol}`;
        return new Promise((resolve, reject) => {
            this.http.get(apiUrl).subscribe(
                (data: any) => {
                    this.stockData = data;
                    this.initializeChart();
                    this.router.navigate(['/stocks'], {queryParams: {symbol: symbol}});
                    // Resolve the promise when the API call is complete
                    resolve();
                },
                (error) => {
                    console.error('Error fetching stock data:', error);
                    this.isLoading = false; // Data loading failed
                    reject(error); // Reject the promise if the API call fails
                }
            );
        });
    }

    getSummary(): string {
        // Signal mapping
        const signalMapping: { [key: string]: number } = {
            'STRONG SELL': -2,
            'SELL': -1,
            'NEUTRAL': 0,
            'BUY': 1,
            'STRONG BUY': 2
        };

        // Get oscillator signal value
        const oscillatorSignalValue = signalMapping[this.oscillatorSIGNAL] ?? 0;

        // Get moving averages signal value
        const maSignalValue = signalMapping[this.maSIGNAL] ?? 0;

        // Calculate the average signal
        const averageSignalValue = (oscillatorSignalValue + maSignalValue) / 2;

        // Determine the summary based on the average signal value
        if (averageSignalValue <= -1.5) {
            return 'STRONG SELL';
        } else if (averageSignalValue > -1.5 && averageSignalValue <= -0.5) {
            return 'SELL';
        } else if (averageSignalValue > -0.5 && averageSignalValue <= 0.5) {
            return 'NEUTRAL';
        } else if (averageSignalValue > 0.5 && averageSignalValue <= 1.5) {
            return 'BUY';
        } else {
            return 'STRONG BUY';
        }
    }

    getGaugeValue(label: string): number {
        const signalMapping: { [key: string]: number } = {
            'STRONG SELL': 10,
            'SELL': 30,
            'NEUTRAL': 50,
            'BUY': 70,
            'STRONG BUY': 90
        };

        return signalMapping[label];
    }

    initializeChart(): void {
        this.companyName = this.stockData['Company Name'];
        this.currentPrice = this.stockData['Current Price'];
        this.allTimeHigh = this.stockData['MAX Price'];
        this.allTimeLow = this.stockData['MIN Price'];
        this.revenueGenerated = this.stockData['REVENUE'];
        this.averagePrice = this.stockData['AVERAGE PRICE'];

        this.rsiValue = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['RSI'].toFixed(2);
        this.macdValue = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['MACD'].toFixed(2);
        this.stochValue = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['Stochastic Oscillator'].toFixed(2);
        this.williamsValue = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['Williams %R'].toFixed(2);
        this.rateOfChangeValue = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['Rate of Change'].toFixed(2);

        this.sma10 = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['SMA10'].toFixed(2);
        this.sma20 = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['SMA20'].toFixed(2);
        this.sma50 = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['SMA50'].toFixed(2);
        this.ema10 = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['EMA10'].toFixed(2);
        this.ema20 = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['EMA20'].toFixed(2);

        this.oscillatorSIGNAL = this.stockData['Timeframes'][`${this.timeFrame}`]['Oscillators']['METER'];
        this.maSIGNAL = this.stockData['Timeframes'][`${this.timeFrame}`]['Moving Averages']['METER'];

        this.pricePredictionFromModel = this.stockData['Timeframes'][`${this.timeFrame}`]['Price Prediction'];
        this.buyOrSellPredictionFromModel = this.stockData['Timeframes'][`${this.timeFrame}`]['Market News Evaluation'];


        const graphData = this.stockData['Timeframes'][`${this.timeFrame}`]['GraphData'].map((entry: {
            DATE: string;
            "PRICE OF LAST TRANSACTION": number;
        }) => {
            const date = new Date(entry.DATE); // Parse the ISO 8601 string into a Date object
            return [
                Date.UTC(
                    date.getUTCFullYear(), // Year
                    date.getUTCMonth(),    // Month (0-based)
                    date.getUTCDate()      // Day
                ),
                entry["PRICE OF LAST TRANSACTION"]
            ];
        });



        this.chartOptions = {
            chart: {
                reflow: true,
                zooming: {
                    type: 'x', // Zoom only on the x-axis
                    resetButton: {
                        position: {
                            align: 'right',
                            verticalAlign: 'top'
                        }
                    }
                },
            },
            title: {
                text: this.companyName
            },
            xAxis: {
                type: 'datetime',
            },
            yAxis: {
                title: {
                    text: 'Price (MKD)'
                }
            },
            series: [{
                type: 'line',
                name: 'Price',
                data: graphData,
                marker: {
                    enabled: false,
                    radius: 4
                }
            }],
            credits: {
                enabled: false // Disable the Highcharts branding
            },
            tooltip: {
                enabled: true, // Tooltip still enabled to display data
                shared: true // Show data details on hover
            }
        };
    }

    changeTimeFrame(time: string) {
        this.timeFrame = time;
        this.initializeChart();
    }

    goBack() {
        this.router.navigate(['/search']);
    }
}
