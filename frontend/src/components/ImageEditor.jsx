import { useState, useEffect, useRef, useMemo } from 'react';
import { LANGS, LangContext, translate } from '../lib/i18n';
import { X, Download, Image, CaretDown, SidebarSimple, ArrowLeft, ArrowRight, Trash, Eye, Intersect } from '@phosphor-icons/react';
import boxIconRaw from '../assets/box.svg?raw'
import { apiPost, apiGet, apiDelete, apiEventSource } from '../lib/api';
import { loadState, saveState, clearState } from '../lib/persist';
import editMode from './modes/Edit';
import removeBackgroundMode from './modes/RemoveBackground';
import flux2KleinMode from './modes/Flux2Klein';
import identityMode from './modes/Identity';

const MODES = {
  'edit': editMode,
  'remove-background': removeBackgroundMode,
  'flux2klein': flux2KleinMode,
  'identity': identityMode,
};

export default function ImageEditor() {
  const [images, setImages] = useState([]);
  const [result, setResult] = useState(null);
  const [resultMeta, setResultMeta] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressBar, setProgressBar] = useState({ target: 0, duration: 200 });
  const [progressMessage, setProgressMessage] = useState('');
  const [remaining, setRemaining] = useState('');
  const lastTickRef = useRef(null);
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
  const [dark, setDark] = useState(() => localStorage.getItem('imgbox:theme') === 'dark');
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
  const menuRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('imgbox:theme', dark ? 'dark' : 'light');
  }, [dark]);

  const modeConfig = MODES[mode];
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
      const merged = { ...cfg.initialState, ...(saved.modeState || {}) };
      setModeState(cfg.restoreState ? cfg.restoreState(merged) : merged);
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
      saveState(mode, { modeState: newState ?? modeState, images, result: blob, meta })
        .catch(err => console.error('saveState:', err));
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

  const Inputs = modeConfig.Inputs;

  return (
    <LangContext.Provider value={langCtx}>
    <div className="relative h-screen bg-white dark:bg-zinc-900 text-gray-900 dark:text-gray-100 flex flex-col p-5 gap-3 overflow-hidden">

      {/* Header */}
      <div className="flex items-baseline gap-2 flex-shrink-0 pl-2 w-full">
        <h1 className="text-2xl font-bold">imgbox</h1>
        <span className="w-6 h-6 block self-center dark:invert" dangerouslySetInnerHTML={{ __html: boxIconRaw.replace(/width="\d+" height="\d+"/, 'width="24" height="24"') }} />
        <div className="relative ml-2" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(o => !o)}
            className="flex items-center gap-2 px-4 py-1.5 text-base font-normal leading-none border border-gray-300 dark:border-zinc-600 rounded hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors min-w-[180px] justify-between"
            style={{ textBox: 'trim-both cap alphabetic' }}
          >
            {t(modeConfig.label)}
            <CaretDown size={14} className={`transition-transform ${menuOpen ? 'rotate-180' : ''}`} />
          </button>
          {menuOpen && (
            <div className="absolute top-full left-0 mt-1 bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 rounded shadow-lg z-50 min-w-[280px] py-1">
              {Object.entries(MODES).map(([value, cfg]) => (
                <button
                  key={value}
                  onClick={() => handleModeChange(value)}
                  className={`w-full text-left px-4 py-2 text-base font-normal transition-colors ${!databaseOpen && mode === value ? 'bg-gray-100 dark:bg-zinc-700' : 'hover:bg-gray-50 dark:hover:bg-zinc-700'}`}
                >
                  {t(cfg.label)}
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => setDatabaseOpen(o => !o)}
          className={`px-4 py-1.5 text-base font-normal leading-none border border-gray-300 dark:border-zinc-600 rounded transition-colors ${databaseOpen ? 'bg-gray-100 dark:bg-zinc-800' : 'hover:bg-gray-50 dark:hover:bg-zinc-800'}`}
          style={{ textBox: 'trim-both cap alphabetic' }}
        >
          {t('common.database')}
        </button>
        <div className="ml-auto self-center" onMouseEnter={() => setSettingsOpen(true)} onMouseLeave={() => setSettingsOpen(false)}>
          <button className="p-1.5 text-black dark:text-gray-100 hover:text-gray-500 dark:hover:text-gray-400 transition-colors">
            <SidebarSimple size={20} mirrored />
          </button>
          {settingsOpen && (
            <div className="absolute top-0 right-0 h-full w-96 bg-white dark:bg-zinc-900 border-l border-gray-200 dark:border-zinc-700 shadow-lg flex flex-col z-50">
              <div className="px-5 pt-5 pb-3 border-b border-gray-100 dark:border-zinc-800 font-semibold">{t('common.settings')}</div>
              <div className="flex flex-col py-2">
                <button
                  onClick={() => { setHfExpanded(o => !o); setHfSaved(false); }}
                  className="text-left px-5 py-3 text-sm hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
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
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 dark:placeholder-gray-500 rounded focus:outline-none focus:border-gray-400 dark:focus:border-zinc-400"
                    />
                    {hfSaved && <span className="text-xs text-green-500 mt-1 block">{t('common.saved')}</span>}
                  </div>
                )}
                <button
                  onClick={() => setLangExpanded(o => !o)}
                  className="text-left px-5 py-3 text-sm hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
                >
                  {t('common.language')}
                </button>
                {langExpanded && (
                  <div className="flex flex-col">
                    {LANGS.map(l => (
                      <button
                        key={l}
                        onClick={() => handleLangChange(l)}
                        className={`text-left px-8 py-2 text-sm transition-colors ${lang === l ? 'text-black dark:text-gray-100 font-medium' : 'text-gray-500 hover:bg-gray-50 dark:hover:bg-zinc-800'}`}
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                )}
                <button
                  onClick={() => setDark(d => !d)}
                  className="flex items-center justify-between px-5 py-3 text-sm hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
                >
                  {t('common.dark_mode')}
                  <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${dark ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`}>
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${dark ? 'translate-x-4' : 'translate-x-0.5'}`} />
                  </span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main row — two cards, or single database card */}
      <div className="flex gap-4 flex-1 min-h-0">

      {databaseOpen ? (
        <div className="flex-1 flex flex-col rounded border border-gray-200 dark:border-zinc-700 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-100 dark:border-zinc-800 flex items-center gap-4 text-xs font-medium text-gray-400 uppercase tracking-wide">
            <div className="w-16 flex-shrink-0">{t('common.picture')}</div>
            <div className="w-48 min-w-0">{t('common.filename')}</div>
            <div className="w-24">{t('common.date')}</div>
            <div className="w-16">{t('common.gender')}</div>
            <div className="flex-1 min-w-0">{t('common.caption')}</div>
            <div className="w-6" />
          </div>
          <div className="flex-1 overflow-y-auto bg-white dark:bg-zinc-900">
            {databaseRows === null ? (
              <div className="h-full" />
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
                      className={`relative px-5 py-2 border-b border-gray-100 dark:border-zinc-800 cursor-pointer ${selected ? 'bg-gray-100 dark:bg-zinc-800' : 'hover:bg-gray-50 dark:hover:bg-zinc-800/60'}`}
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
                        <div className="w-48 min-w-0 text-sm text-gray-700 dark:text-gray-200 truncate" title={filename}>{filename}</div>
                        <div className="text-xs text-gray-400 w-24">{date}</div>
                        <div className="text-xs text-gray-400 w-16">{gender}</div>
                        <div className="flex-1 min-w-0 text-xs text-gray-400 truncate" title={row.caption || ''}>{row.caption || '—'}</div>
                      </div>
                      {selected && (
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center gap-2">
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setSeeRow(row); }}
                            className="pointer-events-auto inline-flex items-center gap-1.5 px-3 py-1 text-xs bg-white dark:bg-zinc-800 border border-gray-300 dark:border-zinc-600 text-gray-600 dark:text-gray-300 rounded shadow-sm hover:bg-gray-800 hover:text-white hover:border-gray-800 dark:hover:bg-gray-100 dark:hover:text-zinc-900 dark:hover:border-gray-100 transition-colors"
                          >
                            <Eye size={14} />
                            {t('common.view')}
                          </button>
                          <button
                            type="button"
                            disabled={matchingId === row.id}
                            onClick={(e) => { e.stopPropagation(); handleMatchIdentity(row.id); }}
                            className="pointer-events-auto inline-flex items-center gap-1.5 px-3 py-1 text-xs bg-white dark:bg-zinc-800 border border-gray-300 dark:border-zinc-600 text-gray-600 dark:text-gray-300 rounded shadow-sm hover:bg-gray-800 hover:text-white hover:border-gray-800 dark:hover:bg-gray-100 dark:hover:text-zinc-900 dark:hover:border-gray-100 transition-colors disabled:cursor-wait"
                          >
                            <Intersect size={14} />
                            {t('common.match')}
                          </button>
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); handleDeleteIdentity(row.id); }}
                            className="pointer-events-auto inline-flex items-center gap-1.5 px-3 py-1 text-xs bg-white dark:bg-zinc-800 border border-red-300 dark:border-red-500/50 text-red-500 dark:text-red-400 rounded shadow-sm hover:bg-red-500 hover:text-white hover:border-red-500 dark:hover:bg-red-500 dark:hover:text-white transition-colors"
                          >
                            <Trash size={14} />
                            {t('common.delete')}
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
        <div className="flex-1 flex flex-col rounded border border-gray-200 dark:border-zinc-700 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-gray-100 dark:border-zinc-800 flex items-center justify-between">
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

          {/* Mode-specific inputs (modes render their own ImageDropZone) */}
          <div className="flex-1 flex flex-col min-h-0">
            <Inputs
              state={modeState}
              setState={setModeState}
              images={images}
              setImages={setImages}
              onZoom={setLightbox}
            />
          </div>

          {/* Shared action buttons */}
          <div className="flex items-center gap-3 px-5 py-4 border-t border-gray-100 dark:border-zinc-800">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 h-9 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
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
                {isLoading ? t('progress.' + (progressMessage || 'Loading')) : t('common.run')}
                {isLoading && remaining && (
                  <span className="opacity-70">· {remaining} {t('common.left')}</span>
                )}
              </span>
            </button>
          </div>
        </div>

        {/* Result card */}
        <div className="flex-1 flex flex-col rounded border border-gray-200 dark:border-zinc-700 overflow-hidden">
          <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100 dark:border-zinc-800">
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
                className="flex items-center px-2 py-1 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
              >
                <Download size={13} />
              </button>
            )}
          </div>

          <div className="flex-1 flex items-center justify-center p-4 bg-gray-50 dark:bg-zinc-800/50 overflow-hidden">
            {modeConfig.Result ? (
              <modeConfig.Result result={result} meta={resultMeta} onZoom={setLightbox} />
            ) : result ? (
              <img src={result} alt="Generated" onClick={() => setLightbox(result)} className="w-full h-full object-contain rounded cursor-zoom-in" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-300 dark:text-zinc-600">
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
        return (
          <div
            onClick={() => setSeeRow(null)}
            className="fixed inset-0 z-[90] bg-black/70 flex items-center justify-center p-8 cursor-pointer"
          >
            <div onClick={(e) => e.stopPropagation()} className="relative bg-white dark:bg-zinc-900 rounded-lg p-6 max-w-2xl w-full cursor-default">
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
                    <div className="text-sm text-gray-700 dark:text-gray-200 break-all">{filename}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.date')}</div>
                    <div className="text-sm text-gray-700 dark:text-gray-200">{date}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.gender')}</div>
                    <div className="text-sm text-gray-700 dark:text-gray-200">{gender}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">{t('common.caption')}</div>
                    <div className="text-sm text-gray-700 dark:text-gray-200 whitespace-pre-wrap break-words">{seeRow.caption || '—'}</div>
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
            <div onClick={(e) => e.stopPropagation()} className="relative bg-white dark:bg-zinc-900 rounded-lg p-6 max-w-3xl w-full cursor-default flex flex-col items-center gap-4">
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
