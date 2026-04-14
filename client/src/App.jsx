import { useState, useEffect } from 'react'
import ImageEditor from './components/ImageEditor'
import Setup from './components/Setup'

function App() {
  const [showSetup, setShowSetup] = useState(false);

  useEffect(() => {
    import('@tauri-apps/api/core').then(({ invoke }) => {
      invoke('get_server_config').then((config) => {
        if (!config.hf_token) setShowSetup(true);
      }).catch(() => {});
    }).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-white">
      {showSetup ? <Setup onDone={() => setShowSetup(false)} /> : <ImageEditor />}
    </div>
  )
}

export default App
