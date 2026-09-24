import { App } from './app/App';

const app = new App(document.getElementById('view') as HTMLCanvasElement);
app.start().catch((err: unknown) => app.showError(err));
