import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';
import 'zone.js'

// Disable long stack traces in production to improve performance
(window as any).__Zone_disable_longStackTrace = true;

// Ignore properties like `setTimeout` and `setInterval`
(window as any).__Zone_ignore_on_properties = [
    { target: window, ignoreProperties: ['setTimeout', 'setInterval'] }
];


bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));


