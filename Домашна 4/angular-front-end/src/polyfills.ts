/***************************************************************************************************
 * Zone.js customizations to handle long tasks
 */

// Disable long task detection
(window as any).__Zone_disable_longStackTrace = true;
(window as any).__Zone_disable_error = true;

// Ignore timeout for Zone.js tracking
(window as any).__Zone_ignore_on_properties = [
    { target: window, ignoreProperties: ['timeout', 'setTimeout'] }
];