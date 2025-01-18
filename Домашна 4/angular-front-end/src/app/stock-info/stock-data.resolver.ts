import { Injectable } from '@angular/core';
import { Resolve, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class StockDataResolver implements Resolve<any> {
    constructor(private http: HttpClient) {}

    resolve(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<any> {
        const symbol = route.queryParams['symbol']; // Get the symbol from query params
        const apiUrl = `https://domasno-das-api.vercel.app/stock-data/${symbol}`;
        return this.http.get(apiUrl); // Return the stock data
    }
}