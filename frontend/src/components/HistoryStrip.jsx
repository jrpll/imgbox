import { useState, useEffect } from 'react';

// Thumbnails of a mode's past results. An entry shows its blob, or the first image of a batch.
export default function HistoryStrip({ entries, shown, onPick }) {
  const [urls, setUrls] = useState([]);

  useEffect(() => {
    const created = entries.map((e) => URL.createObjectURL(e.blob ?? e.meta.images[0]));
    setUrls(created);
    return () => created.forEach(URL.revokeObjectURL);
  }, [entries]);

  return (
    <div className="flex gap-2 px-5 py-3 border-t border-gray-100 dark:border-zinc-800 overflow-x-auto">
      {urls.map((url, i) => (
        <img
          key={url}
          src={url}
          onClick={() => onPick(i)}
          className={`h-12 w-12 flex-shrink-0 object-cover rounded cursor-pointer transition-opacity ${
            i === shown ? 'ring-2 ring-[#0ea0ff]' : 'opacity-60 hover:opacity-100'
          }`}
        />
      ))}
    </div>
  );
}
