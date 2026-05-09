import { useState } from 'react';

export default function Setup({ onDone }) {
  const [token, setToken] = useState('');
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!token.trim()) return;
    setSaving(true);
    const { invoke } = await import('@tauri-apps/api/core');
    const current = await invoke('get_server_config');
    await invoke('set_server_config', {
      config: { ...current, hf_token: token.trim() }
    });
    onDone();
  };

  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <div className="flex flex-col gap-4 w-96">
        <h2 className="text-2xl font-bold">setup</h2>
        <p className="text-gray-500 text-sm">
          A HuggingFace token is required to download the model on first launch.
        </p>
        <input
          type="password"
          placeholder="hf_..."
          value={token}
          onChange={(e) => setToken(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          className="h-10 px-3 border border-gray-500 rounded-lg text-base focus:outline-none"
          autoFocus
        />
        <button
          onClick={handleSave}
          disabled={!token.trim() || saving}
          className="h-10 px-4 rounded-md text-base font-medium text-white bg-[#0ea0ff] hover:bg-blue-900 disabled:opacity-40"
        >
          {saving ? 'saving...' : 'continue'}
        </button>
      </div>
    </div>
  );
}
