import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import { I18nProvider } from './i18n/I18nProvider';
import { App } from './App';
import './styles/app.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <I18nProvider initialLocale="en">
      <HashRouter>
        <App />
      </HashRouter>
    </I18nProvider>
  </React.StrictMode>
);
