import { Routes } from '@angular/router';
import {HomeComponent} from "./home/home.component";
import {SearchComponent} from "./search/search.component";
import {StockInfoComponent} from "./stock-info/stock-info.component";

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'search', component: SearchComponent},
  { path: 'stocks', component: StockInfoComponent}
];
