import { useState, useEffect, useRef, useMemo } from 'react';
import { LANGS, LangContext, translate } from '../lib/i18n';
import { X, Download, Image, CaretDown, SidebarSimple, ArrowBendUpRight, Swap, Command, KeyReturn } from '@phosphor-icons/react';
import boxIconRaw from '../assets/box.svg?raw'
import { apiPost, apiGet, apiDelete, apiJob, apiResult, apiEventSource } from '../lib/api';
import { loadState, saveState, updateState, clearState } from '../lib/persist';
import HistoryStrip from './HistoryStrip';
import ModelsPanel from './ModelsPanel';
import editMode from './modes/Edit';
import removeBackgroundMode from './modes/RemoveBackground';
import sam3Mode from './modes/Sam3';
import flux2KleinMode from './modes/Flux2Klein';
import flux2KleinFastMode from './modes/Flux2KleinFast';
import sdeditMode from './modes/SDEdit';
import identityMode from './modes/Identity';

const MODES = {
  'edit': editMode,
  'remove-background': removeBackgroundMode,
  'sam3': sam3Mode,
  'flux2klein': flux2KleinMode,
  'flux2klein-fast': flux2KleinFastMode,
  'sdedit': sdeditMode,
  'identity': identityMode,
};

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/.test(navigator.platform);
const HISTORY_MAX = 12;
// The stickers beside a mode's name say what the mode is for; one tint per task.
const TASK_TINT = {
  edit: 'bg-blue-50 text-blue-600 dark:bg-blue-500/15 dark:text-blue-300',
  generate: 'bg-violet-50 text-violet-600 dark:bg-violet-500/15 dark:text-violet-300',
  cutout: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300',
  faces: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
};

export default function ImageEditor() {
  const [images, setImages] = useState([]);
  const [result, setResult] = useState(null);
  const [resultMeta, setResultMeta] = useState(null);
  // Runs in flight, by mode: a run keeps going while another mode is open. The value is the
  // server job's id, or true while there is none (the upload, a model load). The ref serves
  // the callbacks that outlive the render they were made in.
  const [running, setRunning] = useState({});
  const runningRef = useRef(running);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressBar, setProgressBar] = useState({ target: 0, duration: 200 });
  const [progressMessage, setProgressMessage] = useState('');
  const [remaining, setRemaining] = useState('');
  const lastTickRef = useRef(null);
  const [mode, setMode] = useState(() => {
    const saved = localStorage.getItem('imgbox:lastMode');
    return saved && MODES[saved] ? saved : 'edit';
  });
  const modeRef = useRef(mode);
  modeRef.current = mode;
  // Counts, per mode, the runs that have ended: a hydration that read IndexedDB just before
  // one did must not put that stale read on screen over what the run brought.
  const endedRef = useRef({});
  const [modeState, setModeState] = useState(MODES[mode].initialState);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelsExpanded, setModelsExpanded] = useState(false);
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
  const [error, setError] = useState(null);
  // This mode's past results, oldest first; `shown` indexes the one on display.
  const [history, setHistory] = useState([]);
  const [shown, setShown] = useState(null);
  const [showSource, setShowSource] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);
  const isLoading = !!running[mode];
  // The open mode's job id, once the server has handed one out: what Cancel cancels.
  const job = typeof running[mode] === 'string' ? running[mode] : null;
  const jobRef = useRef(null);
  jobRef.current = job;
  const setRun = (m, value) => {
    const next = { ...runningRef.current };
    if (value) next[m] = value;
    else delete next[m];
    runningRef.current = next;
    setRunning(next);
  };
  // Chaining: a result image waiting for the mode it goes to (`sending`), then for that mode to open.
  const [sending, setSending] = useState(null);
  const sentRef = useRef(null);

  const t = useMemo(() => (key, params) => translate(key, lang, params), [lang]);
  const handleLangChange = (l) => {
    setLang(l);
    localStorage.setItem('imgbox:lang', l);
    setLangExpanded(false);
  };
  const langCtx = useMemo(() => ({ lang, t, setLang: handleLangChange }), [lang, t]);
  const menuRef = useRef(null);
  const settingsRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('imgbox:theme', dark ? 'dark' : 'light');
  }, [dark]);

  // Scrollbar thumbs are transparent until an area is actually scrolled (see index.css);
  // drop the marker once it has been still for a beat so the bar fades back out.
  useEffect(() => {
    const timers = new WeakMap();
    const onScroll = (e) => {
      const el = e.target;
      if (!(el instanceof Element)) return;
      el.setAttribute('data-scrolling', '');
      clearTimeout(timers.get(el));
      timers.set(el, setTimeout(() => el.removeAttribute('data-scrolling'), 900));
    };
    document.addEventListener('scroll', onScroll, true);
    return () => document.removeEventListener('scroll', onScroll, true);
  }, []);

  const modeConfig = MODES[mode];
  // Every mode names its registry `model` (the header says whether it is loaded). Those with
  // `loadFirst` load it explicitly — until then the Run button is Load model; the others on their first run.
  const needsLoad = !!modeConfig.loadFirst && !modelLoaded;
  const canRun = !isLoading && (needsLoad || modeConfig.canSubmit({ images, state: modeState }));

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
      if (settingsRef.current && !settingsRef.current.contains(e.target)) setSettingsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // The "before" of the result card's before/after toggle.
  useEffect(() => {
    if (!images.length) { setSourceUrl(null); return; }
    const url = URL.createObjectURL(images[0]);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [images]);

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
          // Progress is the running job's. If that isn't ours, ours is waiting in line behind it.
          const queued = !!data.job && !!jobRef.current && data.job !== jobRef.current;
          const p = queued ? 0 : (data.progress ?? 0);
          const msg = queued ? 'Queued' : (data.message ?? '');
          setProgress(p);
          if (msg) setProgressMessage(msg);
          setRemaining(queued ? '' : (data.remaining ?? ''));

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

  const hydratedRef = useRef(false);
  useEffect(() => {
    hydratedRef.current = false;
    let cancelled = false;
    // False once a run of this mode has ended since: what it brought is newer than what is read here.
    const ended = endedRef.current[mode];
    const fresh = () => endedRef.current[mode] === ended;
    (async () => {
      const cfg = MODES[mode];
      const saved = await loadState(mode);
      const savedHistory = (await loadState(`${mode}:history`)) || [];
      if (cancelled) return;
      // An image sent here from a result (see handleSendTo).
      const sent = sentRef.current;
      sentRef.current = null;
      if (fresh()) {
        setHistory(savedHistory);
        setShown(savedHistory.length ? savedHistory.length - 1 : null);
      }
      let state = cfg.initialState;
      let inputs = [];
      if (saved) {
        const merged = { ...cfg.initialState, ...(saved.modeState || {}) };
        // restoreState gets /health: some state only holds while the server still has what it
        // refers to — which it never does for an image that has just been sent in.
        const health = !cfg.restoreState || sent ? {} : await apiGet('/health').catch(() => ({}));
        if (cancelled) return;
        state = cfg.restoreState ? cfg.restoreState(merged, health) : merged;
        if (fresh()) setModeState(state);
        const restored = saved.images ?? (saved.image ? [saved.image] : []);
        // Re-read into memory: a persisted File can still be a stale pointer to a file on disk.
        inputs = await Promise.all(
          restored.map(async (f) => new File([await f.arrayBuffer()], f.name ?? 'image', { type: f.type }))
        );
        if (cancelled) return;
        // The last result belonged to the image a sent one replaces — unless it is only added.
        if (fresh() && (!sent || cfg.multi)) {
          if (saved.result) setResult(URL.createObjectURL(saved.result));
          if (saved.meta) setResultMeta(saved.meta);
          if (saved.error) setError(saved.error);
        }
      }
      if (sent) inputs = cfg.multi ? [...inputs, sent] : [sent];
      if (inputs.length) setImages(inputs);
      hydratedRef.current = true;
      // A run that was still going when the page was left — as opposed to one this page is
      // already waiting on, behind another mode: pick it up where it is.
      if (saved?.job && !runningRef.current[mode]) execute({ images: inputs, state, jobId: saved.job });
    })();
    return () => { cancelled = true; };
  }, [mode]);

  // Whether this mode's model is the one on the GPU. The registry holds a single one, so a
  // run starting or ending — in any mode — can change the answer.
  useEffect(() => {
    let cancelled = false;
    apiGet('/health')
      .then((h) => { if (!cancelled) setModelLoaded(h.model === MODES[mode].model); })
      .catch((err) => console.error('health:', err));
    return () => { cancelled = true; };
  }, [mode, running]);

  // Persist parameter tweaks without waiting for a Run; merge so images/result stay intact
  useEffect(() => {
    if (!hydratedRef.current) return;
    const id = setTimeout(() => {
      updateState(mode, { modeState }).catch(err => console.error('saveState:', err));
    }, 500);
    return () => clearTimeout(id);
  }, [mode, modeState]);

  useEffect(() => {
    return () => { if (result) URL.revokeObjectURL(result); };
  }, [result]);

  // Everything that belongs to the mode is reset in the same update that changes it — not in
  // an effect, a render later: a mode's components must never be handed another mode's state,
  // whose shape they don't know (Sam3 reading `state.points` off FlowID's state threw).
  const handleModeChange = (value) => {
    if (value !== mode) {
      setMode(value);
      setModeState(MODES[value].initialState);
      setImages([]);
      setResult(null);
      setResultMeta(null);
      setError(null);
      setHistory([]);
      setShown(null);
      setShowSource(false);
      setModelLoaded(false);
    }
    localStorage.setItem('imgbox:lastMode', value);
    setMenuOpen(false);
  };

  // A run is a server-side job, so it survives a reload: the mode's submit starts one through
  // `run` — or, given `jobId`, waits on the one a previous page load started. It also survives
  // the mode being left: `mode` here is the run's own, and everything it produces is saved
  // under it before any of it is shown, which only happens if that mode is (still, or again)
  // the open one. Inputs are passed in rather than closed over, since on a pick-up this runs
  // from the hydration effect, where the component's state is still the initial one.
  // The previous result stays up while the run is in flight, and after one that fails.
  const execute = async ({ images, state, jobId }) => {
    const open = () => modeRef.current === mode;
    setError(null);
    setRun(mode, jobId ?? true);
    try {
      const run = async (path, body) => {
        const id = jobId ?? await apiJob(path, body);
        setRun(mode, id);
        if (!jobId) await updateState(mode, { modeState: state, images, job: id, error: null });
        return apiResult(id);
      };
      const { blob, state: newState, meta } = await modeConfig.submit({ images, state, run });
      await saveState(mode, { modeState: newState ?? state, images, result: blob, meta });
      // Kept under its own key, so the debounced modeState save doesn't rewrite every blob.
      let next = null;
      if (blob || meta?.images) {
        next = [...((await loadState(`${mode}:history`)) || []), { blob, meta }].slice(-HISTORY_MAX);
        await saveState(`${mode}:history`, next);
      }
      endedRef.current[mode] = (endedRef.current[mode] || 0) + 1;
      if (!open()) return;
      // Only a state the submit changed (Edit moving on to its step 2): the one it was given
      // is where the run started, and the inputs may have moved on since (SDEdit's strokes).
      if (newState && newState !== state) setModeState(newState);
      setResult(blob ? URL.createObjectURL(blob) : null);
      setResultMeta(meta ?? null);
      setShowSource(false);
      if (next) {
        setHistory(next);
        setShown(next.length - 1);
      }
    } catch (err) {
      console.error('Error:', err);
      // Not errors: a cancelled job (409), and one picked up after the server lost it to a restart (404).
      const message = err.status === 409 || (jobId && err.status === 404) ? null : err.message;
      // Stored too, for a mode that isn't open to show it once it is.
      await updateState(mode, { job: null, error: message }).catch(e => console.error('saveState:', e));
      endedRef.current[mode] = (endedRef.current[mode] || 0) + 1;
      if (open()) setError(message);
    } finally {
      setRun(mode, null);
    }
  };

  const handleSubmit = () => execute({ images, state: modeState });

  // Chaining: the picked mode opens with the image as its input — added to it, for a mode
  // that takes several. Its hydration (above) collects the image from sentRef.
  const handleSendTo = (target) => {
    const file = new File([sending], `sent.${sending.type === 'image/png' ? 'png' : 'jpg'}`, { type: sending.type });
    setSending(null);
    if (target !== mode) {
      sentRef.current = file;
      handleModeChange(target);
      return;
    }
    const cfg = MODES[target];
    setImages(cfg.multi ? [...images, file] : [file]);
    if (cfg.restoreState) setModeState((s) => cfg.restoreState(s, {}));
    if (!cfg.multi) {
      setResult(null);
      setResultMeta(null);
    }
  };

  const handleLoad = async () => {
    const open = () => modeRef.current === mode;
    setError(null);
    setRun(mode, true);
    try {
      await apiPost(`/load/${modeConfig.model}`);
      if (open()) setModelLoaded(true);
    } catch (err) {
      console.error('Error:', err);
      if (open()) setError(err.message);
    } finally {
      setRun(mode, null);
    }
  };

  // Modes whose output updates as you interact push it straight up, without a Run step.
  const handleLiveResult = (blob) => {
    setResult(blob ? URL.createObjectURL(blob) : null);
    setResultMeta(null);
    setError(null);
    setShown(null);
    setShowSource(false);
  };

  const handlePickHistory = (i) => {
    const { blob, meta } = history[i];
    setResult(blob ? URL.createObjectURL(blob) : null);
    setResultMeta(meta ?? null);
    setShown(i);
    setShowSource(false);
  };

  const handleReset = async () => {
    setModeState(modeConfig.initialState);
    setImages([]);
    setResult(null);
    setResultMeta(null);
    setError(null);
    setHistory([]);
    setShown(null);
    await clearState(mode);
    await clearState(`${mode}:history`);
  };

  const handleRun = needsLoad ? handleLoad : handleSubmit;

  // Cmd/Ctrl+Enter to run — ref keeps the listener pointed at the latest state without re-binding
  const runRef = useRef({ canRun, handleRun });
  runRef.current = { canRun, handleRun };
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') { setLightbox(null); setSending(null); }
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (runRef.current.canRun) runRef.current.handleRun();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const Inputs = modeConfig.Inputs;
  const onRun = () => { if (runRef.current.canRun) runRef.current.handleRun(); };
  const stickers = (cfg) => (
    <span className="flex items-center gap-1 flex-shrink-0">
      {cfg.tasks.map((task) => (
        <span key={task} className={`px-1.5 py-1 rounded text-[11px] leading-none font-normal ${TASK_TINT[task]}`}>
          {t(`task.${task}`)}
        </span>
      ))}
    </span>
  );
  // Before/after only makes sense for one image in, one image out.
  const canCompare = !!sourceUrl && images.length === 1 && !modeConfig.Result;
  const viewed = showSource && canCompare ? sourceUrl : result;

  return (
    <LangContext.Provider value={langCtx}>
    <div className="relative h-screen bg-white dark:bg-zinc-900 text-gray-800 dark:text-gray-200 flex flex-col p-5 gap-3 overflow-hidden">

      {/* Header */}
      <div className="flex items-center gap-2 flex-shrink-0 pl-2 w-full">
        <span className="w-7 h-7 block self-center text-gray-800 dark:text-gray-200" dangerouslySetInnerHTML={{ __html: boxIconRaw.replace(/width="\d+" height="\d+"/, 'width="28" height="28" fill="currentColor"') }} />
        <div className="relative w-[400px]" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(o => !o)}
            className={`flex items-center gap-2 h-7 px-4 text-base font-normal leading-none border border-gray-300 dark:border-zinc-600 transition-colors w-full ${menuOpen ? 'rounded-t rounded-b-none bg-gray-100 dark:bg-zinc-700' : 'rounded hover:bg-gray-50 dark:hover:bg-zinc-800'}`}
            style={{ textBox: 'trim-both cap alphabetic' }}
          >
            <span className="grid flex-1">
              {Object.entries(MODES).map(([value, cfg]) => (
                <span key={value} className={`col-start-1 row-start-1 text-left ${value === mode ? '' : 'invisible'}`}>{t(cfg.label)}</span>
              ))}
            </span>
            {/* A run going on in a mode other than the open one. */}
            {Object.keys(running).some((m) => m !== mode) && <span className="w-1.5 h-1.5 rounded-full bg-[#0ea0ff] animate-pulse flex-shrink-0" />}
            {stickers(modeConfig)}
            <CaretDown size={14} className={`flex-shrink-0 transition-transform ${menuOpen ? 'rotate-180' : ''}`} />
          </button>
          {menuOpen && (
            <div className="absolute top-full left-0 w-full bg-white dark:bg-zinc-800 border border-t-0 border-gray-300 dark:border-zinc-600 rounded-b shadow-lg z-50 overflow-hidden">
              {Object.entries(MODES).map(([value, cfg]) => (
                <button
                  key={value}
                  onClick={() => handleModeChange(value)}
                  className={`w-full flex items-center gap-2 text-left px-4 py-2 text-base font-normal transition-colors ${mode === value ? 'bg-gray-100 dark:bg-zinc-700' : 'hover:bg-gray-50 dark:hover:bg-zinc-700'}`}
                >
                  <span className="flex-1">{t(cfg.label)}</span>
                  {running[value] && <span className="w-1.5 h-1.5 rounded-full bg-[#0ea0ff] animate-pulse flex-shrink-0" />}
                  {stickers(cfg)}
                </button>
              ))}
            </div>
          )}
        </div>
        {modeConfig.model && (
          <span className={`ml-2 flex items-center gap-1.5 text-sm transition-colors ${modelLoaded ? 'text-[#0ea0ff]' : 'text-gray-400'}`}>
            <span className={`w-2 h-2 rounded-full transition-colors ${modelLoaded ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`} />
            {t(modelLoaded ? 'common.model_loaded' : 'common.model_not_loaded')}
          </span>
        )}
        <div className="ml-auto self-center" ref={settingsRef}>
          <button
            onClick={() => setSettingsOpen(o => !o)}
            title={t('common.settings')}
            className="p-1.5 rounded-full text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-zinc-800 transition-colors"
          >
            <SidebarSimple size={20} mirrored />
          </button>
          {settingsOpen && (
            <div className="absolute top-0 right-0 h-full w-96 bg-white dark:bg-zinc-900 border-l border-gray-200 dark:border-zinc-700 shadow-lg flex flex-col z-50">
              <div className="px-5 pt-5 pb-3 border-b border-gray-100 dark:border-zinc-800 font-semibold">{t('common.settings')}</div>
              <div className="flex flex-col py-2 overflow-y-auto">
                <button
                  onClick={() => setModelsExpanded(o => !o)}
                  className="text-left px-5 py-3 text-sm hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
                >
                  {t('common.models')}
                </button>
                {modelsExpanded && <ModelsPanel modes={MODES} />}
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
                        className={`text-left px-8 py-2 text-sm transition-colors ${lang === l ? 'text-gray-800 dark:text-gray-200 font-medium' : 'text-gray-500 hover:bg-gray-50 dark:hover:bg-zinc-800'}`}
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

      {/* Main row — two cards */}
      <div className="flex gap-3 flex-1 min-h-0">

        {/* Input card */}
        <div className="w-[380px] flex-shrink-0 flex flex-col rounded border border-gray-200 dark:border-zinc-700 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-gray-100 dark:border-zinc-800 flex items-center justify-between">
            <span className="font-semibold">{t('common.input')}</span>
            {modeConfig.steps && (
              <div className="flex items-center gap-1">
                {modeConfig.steps.map((step, i) => (
                  <button
                    key={step.label}
                    type="button"
                    onClick={() => setModeState((s) => ({ ...s, step: i + 1 }))}
                    disabled={!!step.enabled && !step.enabled(modeState)}
                    className={`flex items-center gap-1.5 px-2 py-0.5 text-xs rounded transition-colors disabled:opacity-40 ${
                      modeState.step === i + 1
                        ? 'bg-gray-100 dark:bg-zinc-800 text-gray-800 dark:text-gray-200'
                        : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
                    }`}
                  >
                    <span className="tabular-nums">{i + 1}</span>
                    {t(step.label)}
                  </button>
                ))}
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
              onResult={handleLiveResult}
              onError={setError}
              onRun={onRun}
              loaded={modelLoaded}
            />
          </div>

          {/* Shared action buttons */}
          <div className="flex items-center gap-3 px-5 py-4 border-t border-gray-100 dark:border-zinc-800">
            {/* Reset, or Cancel while a job runs; neither during the upload or a model load. */}
            <button
              type="button"
              onClick={job ? () => apiDelete(`/jobs/${job}`).catch(err => console.error('cancel:', err)) : handleReset}
              disabled={isLoading && !job}
              className={`px-4 h-9 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-[transform,background-color] active:scale-[0.96] disabled:opacity-40`}
            >
              {t(job ? 'common.cancel' : 'common.reset')}
            </button>
            <button
              type="button"
              onClick={handleRun}
              disabled={!canRun}
              className={`relative flex-1 h-9 text-sm font-medium rounded text-white transition-[transform,background-color] after:absolute after:content-[''] after:inset-x-0 after:-inset-y-2 ${
                isLoading
                  ? 'bg-[#bce4ff] cursor-wait'
                  : 'bg-[#0ea0ff] hover:bg-blue-500 active:scale-[0.96] disabled:opacity-40'
              }`}
            >
              {isLoading && (
                <span className="absolute inset-0 overflow-hidden rounded">
                  <span
                    className="absolute inset-y-0 left-0 bg-[#0ea0ff]"
                    style={{ width: `${(progressBar.target * 100).toFixed(1)}%`, transition: `width ${progressBar.duration}ms linear` }}
                  />
                </span>
              )}
              <span className="relative flex h-full items-center justify-center gap-2">
                {isLoading ? t('progress.' + (progressMessage || 'Loading')) : t(needsLoad ? 'common.load_model' : 'common.run')}
                {isLoading && remaining && (
                  <span className="opacity-70">· {remaining} {t('common.left')}</span>
                )}
                {!isLoading && (
                  <span className="inline-flex items-center gap-0.5 opacity-60">
                    {IS_MAC ? <Command size={12} weight="bold" /> : <span className="text-xs font-semibold">Ctrl</span>}
                    <KeyReturn size={14} weight="bold" />
                  </span>
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
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={async () => setSending(await fetch(result).then((r) => r.blob()))}
                  title={t('common.send_to')}
                  className="flex items-center px-2 py-1 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-[transform,background-color] active:scale-[0.96]"
                >
                  <ArrowBendUpRight size={13} />
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    const { type } = await fetch(result).then((r) => r.blob());
                    const link = document.createElement('a');
                    link.href = result;
                    link.download = `generated-image.${type === 'image/png' ? 'png' : 'jpg'}`;
                    link.click();
                  }}
                  title={t('common.download')}
                  className="flex items-center px-2 py-1 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-[transform,background-color] active:scale-[0.96]"
                >
                  <Download size={13} />
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="px-5 py-2 max-h-24 overflow-y-auto border-b border-gray-100 dark:border-zinc-800 text-sm text-red-500 dark:text-red-400 whitespace-pre-wrap break-words">
              {error}
            </div>
          )}

          <div className="flex-1 flex items-center justify-center p-4 bg-gray-50 dark:bg-zinc-800/50 overflow-hidden">
            {modeConfig.Result ? (
              <modeConfig.Result
                result={result}
                meta={resultMeta}
                onZoom={setLightbox}
                onSend={setSending}
                state={modeState}
                setState={setModeState}
                images={images}
                loaded={modelLoaded}
                onResult={handleLiveResult}
              />
            ) : result ? (
              <img src={viewed} alt="Generated" onClick={() => setLightbox(viewed)} className="max-w-full max-h-full object-contain rounded cursor-zoom-in" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-300 dark:text-zinc-600">
                <Image size={40} />
              </div>
            )}
          </div>

          {/* Under the image, next to it rather than across the screen: before/after, and
              whatever the mode tunes its result with. Gone when neither applies. */}
          <div className="flex items-center justify-center gap-6 px-5 py-3 border-t border-gray-100 dark:border-zinc-800 empty:hidden">
            {canCompare && result && (
              <button
                type="button"
                onClick={() => setShowSource(s => !s)}
                className={`flex items-center gap-1.5 px-2 py-1 text-xs border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 transition-[transform,background-color] active:scale-[0.96] ${showSource ? 'bg-gray-100 dark:bg-zinc-800' : 'hover:bg-gray-50 dark:hover:bg-zinc-800'}`}
              >
                <Swap size={13} />
                {t('common.compare')}
              </button>
            )}
            {modeConfig.ResultControls && <modeConfig.ResultControls state={modeState} setState={setModeState} onRun={onRun} />}
          </div>

          {history.length > 1 && <HistoryStrip entries={history} shown={shown} onPick={handlePickHistory} />}
        </div>
      </div>

      {/* Settings panel */}

      {/* send-to picker */}
      {sending && (
        <div
          onClick={() => setSending(null)}
          className="fixed inset-0 z-[90] bg-black/70 flex items-center justify-center p-8 cursor-pointer"
        >
          <div onClick={(e) => e.stopPropagation()} className="bg-white dark:bg-zinc-900 rounded-lg py-2 w-72 cursor-default overflow-hidden">
            <div className="px-4 py-2 text-xs text-gray-400 uppercase tracking-wide">{t('common.send_to')}</div>
            {Object.entries(MODES).map(([value, cfg]) => (
              <button
                key={value}
                onClick={() => handleSendTo(value)}
                className="w-full flex items-center gap-2 text-left px-4 py-2 text-base font-normal transition-colors hover:bg-gray-50 dark:hover:bg-zinc-700"
              >
                <span className="flex-1">{t(cfg.label)}</span>
                {stickers(cfg)}
              </button>
            ))}
          </div>
        </div>
      )}

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
