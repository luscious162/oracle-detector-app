import { Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Main from './pages/Main'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/main" element={<Main />} />
    </Routes>
  )
}

export default App
