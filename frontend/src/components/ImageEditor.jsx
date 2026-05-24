import { useState, useEffect, useRef, useMemo } from 'react';
import { LANGS, LangContext, translate } from '../lib/i18n';
import { Upload, X, Download, Image, CaretDown, SidebarSimple, ArrowLeft, ArrowRight } from '@phosphor-icons/react';
import boxIconRaw from '../assets/box.svg?raw'
import { apiPost, apiEventSource } from '../lib/api';
import { loadState, saveState, clearState } from '../lib/persist';
import editMode from './modes/Edit';
import removeBackgroundMode from './modes/RemoveBackground';
import flux2KleinMode from './modes/Flux2Klein';
import { DotmSquare4 } from './dotmatrix/dotm-square-4';

const MODES = {
  'edit': editMode,
  'remove-background': removeBackgroundMode,
  'flux2klein': flux2KleinMode,
};

export default function ImageEditor() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageAspect, setImageAspect] = useState(null);
  const [result, setResult] = useState(null);
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

  const t = useMemo(() => (key) => translate(key, lang), [lang]);
  const handleLangChange = (l) => {
    setLang(l);
    localStorage.setItem('imgbox:lang', l);
    setLangExpanded(false);
  };
  const langCtx = useMemo(() => ({ lang, t, setLang: handleLangChange }), [lang, t]);
  const fileInputRef = useRef(null);
  const menuRef = useRef(null);

  const modeConfig = MODES[mode];
  const canRun = !isLoading && modeConfig.canSubmit({ image, state: modeState });

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
          setProgress(p);
          if (data.message) setProgressMessage(data.message);
          setRemaining(data.remaining ?? '');

          const now = Date.now();
          const last = lastTickRef.current;
          if (last) {
            const interval = now - last.t;
            const delta = Math.max(0, p - last.p);
            setProgressBar({ target: Math.min(1, p + delta), duration: interval });
          } else {
            setProgressBar({ target: p, duration: 200 });
          }
          lastTickRef.current = { p, t: now };
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
    setImage(null);
    setResult(null);
    let cancelled = false;
    (async () => {
      const saved = await loadState(mode);
      if (cancelled || !saved) return;
      const cfg = MODES[mode];
      setModeState(cfg.restoreState ? cfg.restoreState(saved.modeState) : saved.modeState);
      if (saved.image) setImage(saved.image);
      if (saved.result) setResult(URL.createObjectURL(saved.result));
    })();
    return () => { cancelled = true; };
  }, [mode]);

  useEffect(() => {
    return () => { if (result) URL.revokeObjectURL(result); };
  }, [result]);

  useEffect(() => {
    if (!image) { setImageUrl(null); setImageAspect(null); return; }
    const url = URL.createObjectURL(image);
    setImageUrl(url);
    setImageAspect(null);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const handleModeChange = (value) => {
    setMode(value);
    localStorage.setItem('imgbox:lastMode', value);
    setMenuOpen(false);
  };

  const handleSubmit = async () => {
    setResult(null);
    setIsLoading(true);
    try {
      const { blob, state: newState } = await modeConfig.submit({ image, state: modeState });
      setModeState(newState);
      setResult(URL.createObjectURL(blob));
      await saveState(mode, { modeState: newState, image, result: blob });
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    setModeState(modeConfig.initialState);
    setImage(null);
    setResult(null);
    await clearState(mode);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setImage(file);
    setIsDragging(false);
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
                  className={`w-full text-left px-4 py-2 text-base font-normal transition-colors ${mode === value ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
                >
                  {t(cfg.label)}
                </button>
              ))}
            </div>
          )}
        </div>
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

      {/* Main row — two cards */}
      <div className="flex gap-4 flex-1 min-h-0">

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
            <span className="text-xs text-gray-400 group-hover:text-gray-600">{t('common.image')}</span>
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`relative h-40 flex items-center justify-center rounded cursor-pointer overflow-hidden transition-colors ${
                isDragging ? 'bg-gray-100' : image ? '' : 'bg-gray-50'
              }`}
              style={{
                backgroundImage: isDragging
                  ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                  : isHovered
                    ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                    : image
                      ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23e5e7eb' stroke-width='2' stroke-dasharray='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                      : `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23d1d5db' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
              }}
            >
              {image && imageUrl ? (
                <div
                  className="relative max-w-[calc(100%-24px)] max-h-[calc(100%-24px)]"
                  style={{ aspectRatio: imageAspect ?? 1 }}
                >
                  <img
                    src={imageUrl}
                    alt="Error loading image"
                    onLoad={(e) => setImageAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
                    onClick={(e) => { e.stopPropagation(); setLightbox(imageUrl); }}
                    className="block w-full h-full object-contain rounded cursor-zoom-in"
                  />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setImage(null); setResult(null); }}
                    className="absolute top-1 right-1 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-400">
                  <Upload size={24} />
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(e) => { const f = e.target.files[0]; if (f) setImage(f); }}
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

          <div className="flex-1 flex items-center justify-center p-4 bg-gray-50">
            {result ? (
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
      </div>

      {/* Settings panel */}

      {/* lightbox */}
      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center p-8 cursor-zoom-out"
        >
          <img src={lightbox} alt="" className="max-w-full max-h-full object-contain" onClick={(e) => e.stopPropagation()} />
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
