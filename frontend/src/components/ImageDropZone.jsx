import { useState, useEffect, useRef } from 'react';
import { Upload, X } from '@phosphor-icons/react';
import { useLang } from '../lib/i18n';

export default function ImageDropZone({ images, onChange, multi = false, directory = false, onZoom }) {
  const { t } = useLang();
  const [imageUrls, setImageUrls] = useState([]);
  const [imageAspect, setImageAspect] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!images.length) { setImageUrls([]); setImageAspect(null); return; }
    let cancelled = false;
    const created = [];
    (async () => {
      const urls = await Promise.all(images.map(async (f) => {
        const isHeic = /\.(heic|heif)$/i.test(f.name) || /image\/hei[cf]/i.test(f.type);
        let blob = f;
        if (isHeic) {
          try {
            const mod = await import('heic2any');
            const out = await mod.default({ blob: f, toType: 'image/jpeg', quality: 0.6 });
            blob = Array.isArray(out) ? out[0] : out;
          } catch (e) { console.error('HEIC convert failed', e); }
        }
        const u = URL.createObjectURL(blob);
        created.push(u);
        return u;
      }));
      if (!cancelled) setImageUrls(urls);
    })();
    setImageAspect(null);
    return () => {
      cancelled = true;
      created.forEach(URL.revokeObjectURL);
    };
  }, [images]);

  const addFiles = (incoming) => {
    const fresh = Array.from(incoming).filter(f => f.type.startsWith('image/'));
    if (!fresh.length) return;
    onChange(prev => multi ? [...prev, ...fresh] : fresh.slice(0, 1));
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const items = e.dataTransfer.items;
    if (!items || !items[0]?.webkitGetAsEntry) {
      addFiles(e.dataTransfer.files);
      return;
    }
    const files = [];
    for (const item of items) {
      const entry = item.webkitGetAsEntry();
      if (!entry) continue;
      if (entry.isFile) {
        await new Promise((res) => entry.file((f) => { files.push(f); res(); }, res));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        while (true) {
          const batch = await new Promise((res) => reader.readEntries(res, () => res([])));
          if (!batch.length) break;
          for (const child of batch) {
            if (!child.isFile) continue;
            await new Promise((res) => child.file((f) => { files.push(f); res(); }, res));
          }
        }
      }
    }
    addFiles(files);
  };

  const removeAt = (idx) => onChange(prev => prev.filter((_, i) => i !== idx));

  return (
    <div
      className="group flex flex-col gap-1"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <span className="text-xs text-gray-400 group-hover:text-gray-600">
        {t('common.image')}
        {multi && images.length > 0 && <span className="text-gray-300"> · {images.length}</span>}
      </span>
      <div
        onClick={() => (multi || !images.length) && fileInputRef.current?.click()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`relative h-40 flex items-center justify-center rounded overflow-hidden transition-colors ${
          multi || !images.length ? 'cursor-pointer' : ''
        } ${isDragging ? 'bg-gray-100' : 'bg-gray-50'}`}
        style={{
          backgroundImage: isDragging
            ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
            : isHovered
              ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
              : images.length && !multi
                ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23e5e7eb' stroke-width='2' stroke-dasharray='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                : `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23d1d5db' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
        }}
      >
        {!images.length ? (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <Upload size={24} />
          </div>
        ) : multi ? (
          <div className="w-full h-full flex gap-2 p-2 overflow-x-auto">
            {imageUrls.map((url, i) => (
              <div key={i} className="relative h-full aspect-square flex-shrink-0 group/thumb">
                <img
                  src={url}
                  onClick={(e) => { e.stopPropagation(); onZoom?.(url); }}
                  className="w-full h-full object-cover rounded cursor-zoom-in"
                />
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); removeAt(i); }}
                  className="absolute top-1 right-1 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500 opacity-0 group-hover/thumb:opacity-100 transition-opacity"
                >
                  <X size={10} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div
            className="relative max-w-[calc(100%-24px)] max-h-[calc(100%-24px)]"
            style={{ aspectRatio: imageAspect ?? 1 }}
          >
            <img
              src={imageUrls[0]}
              alt="Error loading image"
              onLoad={(e) => setImageAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
              onClick={(e) => { e.stopPropagation(); onZoom?.(imageUrls[0]); }}
              className="block w-full h-full object-contain rounded cursor-zoom-in"
            />
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); removeAt(0); }}
              className="absolute top-1 right-1 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
            >
              <X size={12} />
            </button>
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="image/*"
        multiple={multi}
        {...(directory ? { webkitdirectory: '', directory: '' } : {})}
        onChange={(e) => { addFiles(e.target.files); e.target.value = ''; }}
      />
    </div>
  );
}
