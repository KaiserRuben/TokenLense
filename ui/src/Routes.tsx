// routes.tsx
import { Routes, Route, Navigate } from 'react-router';
import Overview from './views/Overview';
import Dashboard from './views/Dashboard';

const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/analysis/:id" element={<Dashboard />} />
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    );
};

export default AppRoutes;