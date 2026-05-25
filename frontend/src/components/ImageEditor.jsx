import { useState, useEffect, useRef, useMemo } from 'react';
import { LANGS, LangContext, translate } from '../lib/i18n';
import { Upload, X, Download, Image, CaretDown, SidebarSimple, ArrowLeft, ArrowRight, Trash, Eye, Intersect } from '@phosphor-icons/react';
import boxIconRaw from '../assets/box.svg?raw'
import { apiPost, apiGet, apiDelete, apiEventSource } from '../lib/api';
import { loadState, saveState, clearState } from '../lib/persist';
import editMode from './modes/Edit';
import removeBackgroundMode from './modes/RemoveBackground';
import flux2KleinMode from './modes/Flux2Klein';
import identityMode from './modes/Identity';
import { DotmSquare4 } from './dotmatrix/dotm-square-4';

const MODES = {
  'edit': editMode,
  'remove-background': removeBackgroundMode,
  'flux2klein': flux2KleinMode,
  'identity': identityMode,
};

export default function ImageEditor() {
  const [images, setImages] = useState([]);
  const [imageUrls, setImageUrls] = useState([]);
  const [imageAspect, setImageAspect] = useState(null);
  const [result, setResult] = useState(null);
  const [resultMeta, setResultMeta] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressBar, setProgressBar] = useState({ target: 0, duration: 200 });
  const [progressMessage, setProgressMessage] = useState('');
  const [remaining, setRemaining] = useState('');
  const lastTickRef = useRef(null);
  const [isEditingSlider, setIsEditingSlider] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [mode, setMode] = useState(() => {
    const saved = localStorage.getItem('imgbox:lastMode');
    return saved && MODES[saved] ? saved : 'edit';
  });
  const [modeState, setModeState] = useState(MODES[mode].initialState);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [hfExpanded, setHfExpanded] = useState(false);
  const [hfToken, setHfToken] = useState('');
  const [hfSaved, setHfSaved] = useState(false);
  const [langExpanded, setLangExpanded] = useState(false);
  const [lang, setLang] = useState(() => {
    const saved = localStorage.getItem('imgbox:lang');
    return saved && LANGS.includes(saved) ? saved : 'ENG';
  });
  const [lightbox, setLightbox] = useState(null);
  const [databaseOpen, setDatabaseOpen] = useState(() => localStorage.getItem('imgbox:databaseOpen') === 'true');
  const [databaseRows, setDatabaseRows] = useState(null);
  const [selectedRowId, setSelectedRowId] = useState(null);
  const [matchResult, setMatchResult] = useState(null);
  const [matchIndex, setMatchIndex] = useState(0);
  const [matchingId, setMatchingId] = useState(null);
  const [seeRow, setSeeRow] = useState(null);

  const t = useMemo(() => (key, params) => translate(key, lang, params), [lang]);
  const handleLangChange = (l) => {
    setLang(l);
    localStorage.setItem('imgbox:lang', l);
    setLangExpanded(false);
  };
  const langCtx = useMemo(() => ({ lang, t, setLang: handleLangChange }), [lang, t]);
  const fileInputRef = useRef(null);
  const menuRef = useRef(null);

  const modeConfig = MODES[mode];
  const maxImages = modeConfig.maxImages ?? 1;
  const isMulti = maxImages === 'unlimited' || maxImages > 1;
  const canRun = !isLoading && modeConfig.canSubmit({ images, state: modeState });

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    localStorage.setItem('imgbox:databaseOpen', databaseOpen ? 'true' : 'false');
    if (!databaseOpen) {
      setSelectedRowId(null);
      return;
    }
    let cancelled = false;
    setDatabaseRows(null);
    (async () => {
      try {
        const rows = await apiGet('/identity/list');
        if (!cancelled) setDatabaseRows(rows);
      } catch (e) {
        console.error(e);
        if (!cancelled) setDatabaseRows([]);
      }
    })();
    return () => { cancelled = true; };
  }, [databaseOpen]);

  const handleDeleteIdentity = async (id) => {
    setDatabaseRows(prev => prev ? prev.filter(r => r.id !== id) : prev);
    try {
      await apiDelete(`/identity/${id}`);
    } catch (e) {
      console.error(e);
      try {
        const rows = await apiGet('/identity/list');
        setDatabaseRows(rows);
      } catch {}
    }
  };

  const handleMatchIdentity = async (id) => {
    setMatchingId(id);
    try {
      const matches = await apiGet(`/identity/match/${id}?k=20`);
      const query = databaseRows?.find(r => r.id === id) || { id };
      setMatchIndex(0);
      setMatchResult({ query, matches });
    } catch (e) {
      console.error(e);
    } finally {
      setMatchingId(null);
    }
  };

  useEffect(() => {
    if (!isLoading) {
      setProgress(0);
      setProgressBar({ target: 0, duration: 200 });
      setProgressMessage('');
      setRemaining('');
      lastTickRef.current = null;
      return;
    }
    setProgress(0);
    setProgressBar({ target: 0, duration: 200 });
    setProgressMessage('Loading');
    setRemaining('');
    lastTickRef.current = null;
    let es;
    let cancelled = false;
    (async () => {
      const source = await apiEventSource('/progress');
      if (cancelled) { source.close(); return; }
      es = source;
      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          const p = data.progress ?? 0;
          const msg = data.message ?? '';
          setProgress(p);
          if (msg) setProgressMessage(msg);
          setRemaining(data.remaining ?? '');

          const now = Date.now();
          const last = lastTickRef.current;
          const phaseChanged = last && last.msg !== msg;
          if (!last || phaseChanged) {
            setProgressBar({ target: p, duration: 200 });
          } else {
            const interval = now - last.t;
            const delta = Math.max(0, p - last.p);
            setProgressBar(prev => ({
              target: Math.max(prev.target, Math.min(1, p + delta)),
              duration: interval,
            }));
          }
          lastTickRef.current = { p, t: now, msg };
        } catch {}
      };
    })();
    return () => {
      cancelled = true;
      if (es) es.close();
    };
  }, [isLoading]);

  useEffect(() => {
    setModeState(MODES[mode].initialState);
    setImages([]);
    setResult(null);
    setResultMeta(null);
    let cancelled = false;
    (async () => {
      const saved = await loadState(mode);
      if (cancelled || !saved) return;
      const cfg = MODES[mode];
      setModeState(cfg.restoreState ? cfg.restoreState(saved.modeState) : saved.modeState);
      const restored = saved.images ?? (saved.image ? [saved.image] : []);
      if (restored.length) setImages(restored);
      if (saved.result) setResult(URL.createObjectURL(saved.result));
      if (saved.meta) setResultMeta(saved.meta);
    })();
    return () => { cancelled = true; };
  }, [mode]);

  useEffect(() => {
    return () => { if (result) URL.revokeObjectURL(result); };
  }, [result]);

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

  const handleModeChange = (value) => {
    setMode(value);
    localStorage.setItem('imgbox:lastMode', value);
    setMenuOpen(false);
    setDatabaseOpen(false);
  };

  const handleSubmit = async () => {
    setResult(null);
    setResultMeta(null);
    setIsLoading(true);
    try {
      const { blob, state: newState, meta } = await modeConfig.submit({ images, state: modeState });
      setModeState(newState ?? modeState);
      if (blob) setResult(URL.createObjectURL(blob));
      if (meta) setResultMeta(meta);
      await saveState(mode, { modeState: newState ?? modeState, images, result: blob, meta });
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    setModeState(modeConfig.initialState);
    setImages([]);
    setResult(null);
    setResultMeta(null);
    await clearState(mode);
  };

  const addFiles = (incoming) => {
    const fresh = Array.from(incoming).filter(f => f.type.startsWith('image/'));
    if (!fresh.length) return;
    setImages(prev => {
      const combined = isMulti ? [...prev, ...fresh] : fresh.slice(0, 1);
      return maxImages === 'unlimited' ? combined : combined.slice(0, maxImages);
    });
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

  const removeImageAt = (idx) => {
    setImages(prev => prev.filter((_, i) => i !== idx));
    setResult(null);
  };

  const Inputs = modeConfig.Inputs;

  return (
    <LangContext.Provider value={langCtx}>
    <div className="relative h-screen bg-white flex flex-col p-5 gap-3 overflow-hidden">

      {/* Header */}
      <div className="flex items-baseline gap-2 flex-shrink-0 pl-2 w-full">
        <h1 className="text-2xl font-bold">imgbox</h1>
        <span className="w-6 h-6 block self-center" dangerouslySetInnerHTML={{ __html: boxIconRaw.replace(/width="\d+" height="\d+"/, 'width="24" height="24"') }} />
        <div className="relative ml-2" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(o => !o)}
            className="flex items-center gap-2 px-4 py-1.5 text-base font-normal leading-none border border-gray-300 rounded hover:bg-gray-50 transition-colors min-w-[180px] justify-between"
            style={{ textBox: 'trim-both cap alphabetic' }}
          >
            {t(modeConfig.label)}
            <CaretDown size={14} className={`transition-transform ${menuOpen ? 'rotate-180' : ''}`} />
          </button>
          {menuOpen && (
            <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded shadow-lg z-50 min-w-[280px] py-1">
              {Object.entries(MODES).map(([value, cfg]) => (
                <button
                  key={value}
                  onClick={() => handleModeChange(value)}
                  className={`w-full text-left px-4 py-2 text-base font-normal transition-colors ${!databaseOpen && mode === value ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
                >
                  {t(cfg.label)}
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => setDatabaseOpen(o => !o)}
          className={`px-4 py-1.5 text-base font-normal leading-none border border-gray-300 rounded transition-colors ${databaseOpen ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
          style={{ textBox: 'trim-both cap alphabetic' }}
        >
          {t('common.database')}
        </button>
        <div className="ml-auto self-center" onMouseEnter={() => setSettingsOpen(true)} onMouseLeave={() => setSettingsOpen(false)}>
          <button className="p-1.5 text-black hover:text-gray-500 transition-colors">
            <SidebarSimple size={20} mirrored />
          </button>
          {settingsOpen && (
            <div className="absolute top-0 right-0 h-full w-96 bg-white border-l border-gray-200 shadow-lg flex flex-col z-50">
              <div className="px-5 pt-5 pb-3 border-b border-gray-100 font-semibold">{t('common.settings')}</div>
              <div className="flex flex-col py-2">
                <button
                  onClick={() => { setHfExpanded(o => !o); setHfSaved(false); }}
                  className="text-left px-5 py-3 text-sm hover:bg-gray-50 transition-colors"
                >
                  {t('common.hf_token')}
                </button>
                {hfExpanded && (
                  <div className="px-5 pb-3">
                    <input
                      autoFocus
                      type="password"
                      value={hfToken}
                      onChange={e => { setHfToken(e.target.value); setHfSaved(false); }}
                      onKeyDown={async e => {
                        if (e.key !== 'Enter' || !hfToken.trim()) return;
                        const fd = new FormData();
                        fd.append('hf_token', hfToken.trim());
                        await apiPost('/config', fd);
                        setHfSaved(true);
                      }}
                      placeholder="hf_..."
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:border-gray-400"
                    />
                    {hfSaved && <span className="text-xs text-green-500 mt-1 block">{t('common.saved')}</span>}
                  </div>
                )}
                <button
                  onClick={() => setLangExpanded(o => !o)}
                  className="text-left px-5 py-3 text-sm hover:bg-gray-50 transition-colors"
                >
                  {t('common.language')}
                </button>
                {langExpanded && (
                  <div className="flex flex-col">
                    {LANGS.map(l => (
                      <button
                        key={l}
                        onClick={() => handleLangChange(l)}
                        className={`text-left px-8 py-2 text-sm transition-colors ${lang === l ? 'text-black font-medium' : 'text-gray-500 hover:bg-gray-50'}`}
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main row — two cards, or single database card */}
      <div className="flex gap-4 flex-1 min-h-0">

      {databaseOpen ? (
        <div className="flex-1 flex flex-col rounded border border-gray-200 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-100 flex items-center gap-4 text-xs font-medium text-gray-400 uppercase tracking-wide">
            <div className="w-16 flex-shrink-0">{t('common.picture')}</div>
            <div className="w-48 min-w-0">{t('common.filename')}</div>
            <div className="w-24">{t('common.date')}</div>
            <div className="w-16">{t('common.gender')}</div>
            <div className="flex-1 min-w-0">{t('common.caption')}</div>
            <div className="w-6" />
          </div>
          <div className="flex-1 overflow-y-auto bg-white">
            {databaseRows === null ? (
              <div className="flex items-center justify-center h-full text-gray-300">
                <DotmSquare4 size={24} dotSize={3} />
              </div>
            ) : databaseRows.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                {t('common.empty_database')}
              </div>
            ) : (
              <div>
                {databaseRows.map(row => {
                  const url = `/identity/crop/${row.id}`;
                  const gender = row.gender === 0 ? 'F' : row.gender === 1 ? 'M' : '—';
                  const date = row.created_at ? row.created_at.slice(0, 10) : '—';
                  const filename = (row.source_filename || '').split(/[\\/]/).pop();
                  const selected = selectedRowId === row.id;
                  return (
                    <div
                      key={row.id}
                      onClick={() => setSelectedRowId(selected ? null : row.id)}
                      className={`relative px-5 py-2 border-b border-gray-100 cursor-pointer ${selected ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-16 flex-shrink-0">
                          <img
                            src={url}
                            onClick={(e) => { e.stopPropagation(); setLightbox(url); }}
                            onError={(e) => { e.currentTarget.style.display = 'none'; }}
                            className="w-10 h-10 object-cover rounded cursor-zoom-in bg-gray-100"
                          />
                        </div>
                        <div className="w-48 min-w-0 text-sm text-gray-700 truncate" title={filename}>{filename}</div>
                        <div className="text-xs text-gray-400 w-24">{date}</div>
                        <div className="text-xs text-gray-400 w-16">{gender}</div>
                        <div className="flex-1 min-w-0 text-xs text-gray-400 truncate" title={row.caption || ''}>{row.caption || '—'}</div>
                      </div>
                      {selected && (
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center gap-2">
                          <button
                            type="button"
                            title={t('common.see')}
                            onClick={(e) => { e.stopPropagation(); setSeeRow(row); }}
                            className="pointer-events-auto p-1.5 bg-white border border-gray-300 text-gray-600 rounded shadow-sm hover:bg-gray-800 hover:text-white hover:border-gray-800 transition-colors"
                          >
                            <Eye size={16} />
                          </button>
                          <button
                            type="button"
                            title={t('common.match')}
                            disabled={matchingId === row.id}
                            onClick={(e) => { e.stopPropagation(); handleMatchIdentity(row.id); }}
                            className="pointer-events-auto p-1.5 bg-white border border-gray-300 text-gray-600 rounded shadow-sm hover:bg-gray-800 hover:text-white hover:border-gray-800 transition-colors disabled:cursor-wait"
                          >
                            <Intersect size={16} />
                          </button>
                          <button
                            type="button"
                            title={t('common.delete')}
                            onClick={(e) => { e.stopPropagation(); handleDeleteIdentity(row.id); }}
                            className="pointer-events-auto p-1.5 bg-white border border-red-300 text-red-500 rounded shadow-sm hover:bg-red-500 hover:text-white hover:border-red-500 transition-colors"
                          >
                            <Trash size={16} />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      ) : (<>

        {/* Input card */}
        <div className="flex-1 flex flex-col rounded border border-gray-200 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-gray-100 flex items-center justify-between">
            <span className="font-semibold">
              {t('common.input')}
              {modeConfig.getStepLabel && (
                <span className="text-gray-400 font-normal"> · {modeConfig.getStepLabel(modeState, t)}</span>
              )}
            </span>
            {modeConfig.totalSteps && (
              <div className="flex items-center gap-0.5">
                <button
                  type="button"
                  onClick={() => setModeState((s) => ({ ...s, step: s.step - 1 }))}
                  disabled={modeState.step === 1}
                  className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ArrowLeft size={15} />
                </button>
                <button
                  type="button"
                  onClick={() => setModeState((s) => ({ ...s, step: s.step + 1 }))}
                  disabled={modeState.step === modeConfig.totalSteps}
                  className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ArrowRight size={15} />
                </button>
              </div>
            )}
          </div>

          {/* Image upload — hidden in step 2+ of multi-step modes */}
          <div className={`px-5 pt-4 flex-shrink-0 ${modeConfig.totalSteps && modeState.step > 1 ? 'hidden' : ''}`}>
            <div
              className="group flex flex-col gap-1"
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
            <span className="text-xs text-gray-400 group-hover:text-gray-600">
              {t('common.image')}
              {isMulti && images.length > 0 && <span className="text-gray-300"> · {images.length}</span>}
            </span>
            <div
              onClick={() => (isMulti || !images.length) && fileInputRef.current?.click()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`relative h-40 flex items-center justify-center rounded overflow-hidden transition-colors ${
                isMulti || !images.length ? 'cursor-pointer' : ''
              } ${isDragging ? 'bg-gray-100' : images.length ? '' : 'bg-gray-50'}`}
              style={{
                backgroundImage: isDragging
                  ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                  : isHovered
                    ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                    : images.length && !isMulti
                      ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23e5e7eb' stroke-width='2' stroke-dasharray='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                      : `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23d1d5db' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
              }}
            >
              {!images.length ? (
                <div className="flex flex-col items-center gap-2 text-gray-400">
                  <Upload size={24} />
                </div>
              ) : isMulti ? (
                <div className="w-full h-full flex gap-2 p-2 overflow-x-auto">
                  {imageUrls.map((url, i) => (
                    <div key={i} className="relative h-full aspect-square flex-shrink-0 group/thumb">
                      <img
                        src={url}
                        onClick={(e) => { e.stopPropagation(); setLightbox(url); }}
                        className="w-full h-full object-cover rounded cursor-zoom-in"
                      />
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); removeImageAt(i); }}
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
                    onClick={(e) => { e.stopPropagation(); setLightbox(imageUrls[0]); }}
                    className="block w-full h-full object-contain rounded cursor-zoom-in"
                  />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); removeImageAt(0); }}
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
              multiple={isMulti}
              {...(isMulti ? { webkitdirectory: '', directory: '' } : {})}
              onChange={(e) => { addFiles(e.target.files); e.target.value = ''; }}
            />
            </div>
          </div>

          {/* Mode-specific inputs */}
          <div className="flex-1 flex flex-col min-h-0">
            <Inputs
              state={modeState}
              setState={setModeState}
              result={result}
              onResult={setResult}
              onEditingSlider={setIsEditingSlider}
            />
          </div>

          {/* Shared action buttons */}
          <div className="flex items-center gap-3 px-5 py-4 border-t border-gray-100">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 h-9 text-sm border border-gray-300 rounded text-gray-600 hover:bg-gray-50 transition-colors"
            >
              {t('common.reset')}
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canRun}
              className={`relative overflow-hidden flex-1 h-9 text-sm font-medium rounded text-white transition-colors ${
                isLoading
                  ? 'bg-[#bce4ff] cursor-wait'
                  : 'bg-[#0ea0ff] hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed'
              }`}
            >
              {isLoading && (
                <div
                  className="absolute inset-y-0 left-0 bg-[#0ea0ff]"
                  style={{ width: `${(progressBar.target * 100).toFixed(1)}%`, transition: `width ${progressBar.duration}ms linear` }}
                />
              )}
              <span className="relative flex h-full items-center justify-center gap-2">
                {isLoading && <DotmSquare4 size={16} dotSize={2} />}
                {isLoading ? t('progress.' + (progressMessage || 'Loading')) : t('common.run')}
                {isLoading && remaining && (
                  <span className="opacity-70">· {remaining} {t('common.left')}</span>
                )}
              </span>
            </button>
          </div>
        </div>

        {/* Result card */}
        <div className="flex-1 flex flex-col rounded border border-gray-200 overflow-hidden">
          <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
            <span className="font-semibold">{t('common.result')}</span>
            {result && (
              <button
                type="button"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = result;
                  link.download = 'generated-image.png';
                  link.click();
                }}
                className="flex items-center px-2 py-1 text-sm border border-gray-300 rounded text-gray-600 hover:bg-gray-50 transition-colors"
              >
                <Download size={13} />
              </button>
            )}
          </div>

          <div className="flex-1 flex items-center justify-center p-4 bg-gray-50 overflow-hidden">
            {modeConfig.Result ? (
              <modeConfig.Result result={result} meta={resultMeta} onZoom={setLightbox} />
            ) : result ? (
              <div className="relative w-full h-full">
                {isEditingSlider && (
                  <div className="absolute inset-0 bg-black/25 flex items-center justify-center z-10 rounded">
                    <span className="text-white text-sm">{t('common.updating')}</span>
                  </div>
                )}
                <img src={result} alt="Generated" onClick={() => setLightbox(result)} className="w-full h-full object-contain rounded cursor-zoom-in" />
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-300">
                <Image size={40} />
              </div>
            )}
          </div>
        </div>
      </>)}
      </div>

      {/* Settings panel */}

      {/* see detail overlay */}
      {seeRow && (() => {
        const filename = (seeRow.source_filename || '').split(/[\\/]/).pop() || '—';
        const date = seeRow.created_at ? seeRow.created_at.slice(0, 10) : '—';
        const gender = seeRow.gender === 0 ? 'F' : seeRow.gender === 1 ? 'M' : '—';
        const age = typeof seeRow.age === 'number' && seeRow.age >= 0 ? seeRow.age : '—';
        return (
          <div
            onClick={() => setSeeRow(null)}
            className="fixed inset-0 z-[90] bg-black/70 flex items-center justify-center p-8 cursor-pointer"
          >
            <div onClick={(e) => e.stopPropagation()} className="relative bg-white rounded-lg p-6 max-w-2xl w-full cursor-default">
              <button
                type="button"
                onClick={() => setSeeRow(null)}
                className="absolute top-3 right-3 p-1 text-gray-400 hover:text-gray-600"
              >
                <X size={18} />
              </button>
              <div className="flex gap-6">
                <img
                  src={`/identity/crop/${seeRow.id}`}
                  onClick={() => setLightbox(`/identity/crop/${seeRow.id}`)}
                  className="w-64 h-64 object-cover rounded cursor-zoom-in bg-gray-100 flex-shrink-0"
                />
                <div className="flex-1 min-w-0 flex flex-col gap-3 pt-1">
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.filename')}</div>
                    <div className="text-sm text-gray-700 break-all">{filename}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.date')}</div>
                    <div className="text-sm text-gray-700">{date}</div>
                  </div>
                  <div className="flex gap-6">
                    <div>
                      <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.gender')}</div>
                      <div className="text-sm text-gray-700">{gender}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 uppercase tracking-wide">age</div>
                      <div className="text-sm text-gray-700">{age}</div>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.caption')}</div>
                    <div className="text-sm text-gray-700 whitespace-pre-wrap break-words">{seeRow.caption || '—'}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* match results overlay */}
      {matchResult && (() => {
        const total = matchResult.matches.length;
        const m = total > 0 ? matchResult.matches[matchIndex] : null;
        const fmtName = (r) => (r?.source_filename || '').split(/[\\/]/).pop() || '—';
        const fmtDate = (r) => r?.created_at ? r.created_at.slice(0, 10) : '—';
        const fmtGender = (r) => r?.gender === 0 ? 'F' : r?.gender === 1 ? 'M' : '—';
        return (
          <div
            onClick={() => setMatchResult(null)}
            className="fixed inset-0 z-[90] bg-black/70 flex items-center justify-center p-8 cursor-pointer"
          >
            <div onClick={(e) => e.stopPropagation()} className="relative bg-white rounded-lg p-6 max-w-3xl w-full cursor-default flex flex-col items-center gap-4">
              <button
                type="button"
                onClick={() => setMatchResult(null)}
                className="absolute top-3 right-3 p-1 text-gray-400 hover:text-gray-600"
              >
                <X size={18} />
              </button>
              <div className="flex gap-6 justify-center items-start">
                <div className="flex flex-col items-center gap-1">
                  <img
                    src={`/identity/crop/${matchResult.query.id}`}
                    onClick={() => setLightbox(`/identity/crop/${matchResult.query.id}`)}
                    className="w-40 h-40 object-cover rounded cursor-zoom-in bg-gray-100"
                  />
                  <span className="text-xs text-gray-500 max-w-[160px] truncate" title={fmtName(matchResult.query)}>{fmtName(matchResult.query)}</span>
                  <span className="text-[11px] text-gray-400">{fmtDate(matchResult.query)} · {fmtGender(matchResult.query)}</span>
                </div>
                {m ? (
                  <div className="flex flex-col items-center gap-1">
                    <img
                      src={`/identity/crop/${m.id}`}
                      onClick={() => setLightbox(`/identity/crop/${m.id}`)}
                      className="w-40 h-40 object-cover rounded cursor-zoom-in bg-gray-100"
                    />
                    <span className="text-xs text-gray-500 max-w-[160px] truncate" title={fmtName(m)}>{fmtName(m)}</span>
                    <span className="text-[11px] text-gray-400">{fmtDate(m)} · {fmtGender(m)}</span>
                  </div>
                ) : (
                  <div className="flex items-center text-gray-400 text-sm">{t('common.empty_database')}</div>
                )}
              </div>
              {m && (
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setMatchIndex(i => Math.max(0, i - 1))}
                    disabled={matchIndex === 0}
                    className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  >
                    <ArrowLeft size={15} />
                  </button>
                  <span className="text-xs text-gray-400 tabular-nums w-12 text-center">{matchIndex + 1} / {total}</span>
                  <button
                    type="button"
                    onClick={() => setMatchIndex(i => Math.min(total - 1, i + 1))}
                    disabled={matchIndex === total - 1}
                    className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  >
                    <ArrowRight size={15} />
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* lightbox */}
      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center p-8 cursor-zoom-out"
        >
          <img src={lightbox} alt="" className="max-w-full max-h-full object-contain" />
          <button
            type="button"
            onClick={() => setLightbox(null)}
            className="absolute top-4 right-4 p-1.5 bg-white/10 text-white rounded-full hover:bg-white/20"
          >
            <X size={18} />
          </button>
        </div>
      )}
    </div>
    </LangContext.Provider>
  );
}
