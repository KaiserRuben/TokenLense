// App.tsx
import './App.css'
import { BrowserRouter } from "react-router";
import AppRoutes from "@/Routes.tsx";
import { AnalysisProvider } from "@/contexts/AnalysisContext";

function App() {
    return (
        <BrowserRouter>
            <AnalysisProvider batchSize={5}>
                <AppRoutes/>
            </AnalysisProvider>
        </BrowserRouter>
    );
}

export default App;