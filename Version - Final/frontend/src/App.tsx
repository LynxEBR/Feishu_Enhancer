import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './components/Layout'
import TasksPage from './pages/TasksPage'
import BusinessKnowledgePage from './pages/BusinessKnowledgePage'
import ReasoningKnowledgePage from './pages/ReasoningKnowledgePage'

function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<TasksPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/business-knowledge" element={<BusinessKnowledgePage />} />
          <Route path="/reasoning-knowledge" element={<ReasoningKnowledgePage />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  )
}

export default App

