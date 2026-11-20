import { useState, useEffect } from 'react';
import { DownloadSimple } from '@phosphor-icons/react';
import { apiGet, apiPost } from '../lib/api';
import { useLang } from '../lib/i18n';

const gb = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;

// The settings panel's list of models: which are on the server's disk, and a way to fetch
// those that aren't. `modes` is the shell's mode map, to name the modes each model serves.
export default function ModelsPanel({ modes }) {
  const { t } = useLang();
  const [models, setModels] = useState(null);
  const [tick, setTick] = useState(0);

  // Read once, then polled for as long as something is downloading.
  useEffect(() => {
    let timer;
    let cancelled = false;
    const poll = async () => {
      try {
        const list = await apiGet('/models');
        if (cancelled) return;
        setModels(list);
        if (list.some((m) => m.status === 'downloading')) timer = setTimeout(poll, 1000);
      } catch (e) {
        console.error('models:', e);
      }
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [tick]);

  const download = async (name) => {
    await apiPost(`/models/${name}/download`).catch((e) => console.error('download:', e));
    setTick((n) => n + 1);
  };

  return (
    <div className="flex flex-col pb-2">
      {(models || []).map((m) => {
        const ready = m.status === 'ready';
        const downloading = m.status === 'downloading';
        const fraction = m.total ? Math.min(1, m.done / m.total) : 0;
        return (
          <div key={m.name} className="px-5 py-2 flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full flex-shrink-0 transition-colors ${ready ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`} />
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate" title={m.label}>{m.label}</div>
                <div className="text-xs text-gray-400 truncate">{m.modes.map((k) => t(modes[k].label)).join(' · ')}</div>
              </div>
              <span className="text-xs text-gray-400 tabular-nums flex-shrink-0">
                {downloading ? `${gb(m.done)} / ${gb(m.total)}` : gb(m.size)}
              </span>
              {!ready && !downloading && (
                <button
                  type="button"
                  onClick={() => download(m.name)}
                  className="flex items-center gap-1.5 px-2 py-1 text-xs border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-[transform,background-color] active:scale-[0.96] flex-shrink-0"
                >
                  <DownloadSimple size={13} />
                  {t('common.download')}
                </button>
              )}
            </div>
            {downloading && (
              <div className="ml-4 h-1.5 rounded-full bg-[#bce4ff] overflow-hidden">
                <div className="h-full bg-[#0ea0ff] transition-[width] duration-1000 ease-linear" style={{ width: `${(fraction * 100).toFixed(1)}%` }} />
              </div>
            )}
            {m.status === 'error' && (
              <div className="ml-4 max-h-20 overflow-y-auto text-xs text-red-500 dark:text-red-400 whitespace-pre-wrap break-words">{m.error}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
