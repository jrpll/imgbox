import { useState, useEffect } from 'react';
import { X, ArrowLeft, ArrowRight, ArrowBendUpRight, Trash, Eye, Intersect } from '@phosphor-icons/react';
import { apiGet, apiDelete } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';

const initialState = { caption: '' };

function Inputs({ state, setState, images, setImages, onZoom }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <ImageDropZone images={images} onChange={setImages} multi directory onZoom={onZoom} />

      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('common.caption')}</span>
        <div className="relative">
          <textarea
            value={state.caption}
            onChange={(e) => set({ caption: e.target.value })}
            rows={3}
            placeholder={t('identity.caption_example')}
            className="input-textarea"
          />
          {state.caption && (
            <button type="button" onClick={() => set({ caption: '' })} className="input-clear-btn">
              <X size={12} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

async function submit({ images, state, run }) {
  const fd = new FormData();
  for (const img of images) fd.append('images', img);
  if (state?.caption) fd.append('caption', state.caption);
  const r = await run('/identity', fd);
  const data = await r.json();
  return { state, meta: data };
}

// The faces database, under the summary of the last encode.
function Result({ meta, onZoom, onSend }) {
  const { t } = useLang();
  const [databaseRows, setDatabaseRows] = useState(null);
  const [search, setSearch] = useState('');
  const [selectedRowId, setSelectedRowId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [matchResult, setMatchResult] = useState(null);
  const [matchIndex, setMatchIndex] = useState(0);
  const [matchingId, setMatchingId] = useState(null);
  const [seeRow, setSeeRow] = useState(null);

  // `meta` changes with every encode, which is when there are new rows to show.
  useEffect(() => {
    let cancelled = false;
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
  }, [meta]);

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

  const q = search.trim().toLowerCase();
  const visibleRows = (databaseRows || []).filter(r => !q || `${r.source_filename} ${r.caption}`.toLowerCase().includes(q));

  return (
    <div className="w-full h-full flex flex-col gap-3 overflow-hidden">
      {meta && (
        <div className="text-sm text-gray-600 dark:text-gray-300 flex-shrink-0">
          {t('identity.summary', meta)}
        </div>
      )}
      <div className="flex-1 min-h-0 flex flex-col rounded border border-gray-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 overflow-hidden">
        <div className="px-5 py-2 border-b border-gray-100 dark:border-zinc-800 flex items-center gap-4 text-xs font-medium text-gray-400 uppercase tracking-wide">
          <div className="w-16 flex-shrink-0">{t('common.picture')}</div>
          <div className="w-48 min-w-0">{t('common.filename')}</div>
          <div className="w-24">{t('common.date')}</div>
          <div className="w-16">{t('common.gender')}</div>
          <div className="flex-1 min-w-0">{t('common.caption')}</div>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('common.search')}
            className="w-48 px-3 py-1 text-xs font-normal normal-case tracking-normal text-gray-800 border border-gray-300 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 dark:placeholder-gray-500 rounded focus:outline-none focus:border-gray-400 dark:focus:border-zinc-400"
          />
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
              {visibleRows.map(row => {
                const url = `/identity/crop/${row.id}`;
                const gender = row.gender === 0 ? 'F' : row.gender === 1 ? 'M' : '—';
                const date = row.created_at ? row.created_at.slice(0, 10) : '—';
                const filename = (row.source_filename || '').split(/[\\/]/).pop();
                const selected = selectedRowId === row.id;
                const confirming = confirmDeleteId === row.id;
                return (
                  <div
                    key={row.id}
                    onClick={() => { setSelectedRowId(selected ? null : row.id); setConfirmDeleteId(null); }}
                    className={`relative px-5 py-2 border-b border-gray-100 dark:border-zinc-800 cursor-pointer ${selected ? 'bg-gray-100 dark:bg-zinc-800' : 'hover:bg-gray-50 dark:hover:bg-zinc-800/60'}`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-16 flex-shrink-0">
                        <img
                          src={url}
                          onClick={(e) => { e.stopPropagation(); onZoom(url); }}
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
                        {onSend && (
                          <button
                            type="button"
                            onClick={async (e) => { e.stopPropagation(); onSend(await fetch(url).then((r) => r.blob())); }}
                            className="pointer-events-auto inline-flex items-center gap-1.5 px-3 py-1 text-xs bg-white dark:bg-zinc-800 border border-gray-300 dark:border-zinc-600 text-gray-600 dark:text-gray-300 rounded shadow-sm hover:bg-gray-800 hover:text-white hover:border-gray-800 dark:hover:bg-gray-100 dark:hover:text-zinc-900 dark:hover:border-gray-100 transition-colors"
                          >
                            <ArrowBendUpRight size={14} />
                            {t('common.send')}
                          </button>
                        )}
                        {/* Deleting removes the files too, so it takes a second click. */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirming) handleDeleteIdentity(row.id);
                            else setConfirmDeleteId(row.id);
                          }}
                          className={`pointer-events-auto inline-flex items-center gap-1.5 px-3 py-1 text-xs border rounded shadow-sm transition-colors ${
                            confirming
                              ? 'bg-red-500 text-white border-red-500'
                              : 'bg-white dark:bg-zinc-800 border-red-300 dark:border-red-500/50 text-red-500 dark:text-red-400 hover:bg-red-500 hover:text-white hover:border-red-500 dark:hover:bg-red-500 dark:hover:text-white'
                          }`}
                        >
                          <Trash size={14} />
                          {t(confirming ? 'common.confirm' : 'common.delete')}
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
                  onClick={() => onZoom(`/identity/crop/${seeRow.id}`)}
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
                    onClick={() => onZoom(`/identity/crop/${matchResult.query.id}`)}
                    className="w-40 h-40 object-cover rounded cursor-zoom-in bg-gray-100"
                  />
                  <span className="text-xs text-gray-500 max-w-[160px] truncate" title={fmtName(matchResult.query)}>{fmtName(matchResult.query)}</span>
                  <span className="text-[11px] text-gray-400">{fmtDate(matchResult.query)} · {fmtGender(matchResult.query)}</span>
                </div>
                {m ? (
                  <div className="flex flex-col items-center gap-1">
                    <img
                      src={`/identity/crop/${m.id}`}
                      onClick={() => onZoom(`/identity/crop/${m.id}`)}
                      className="w-40 h-40 object-cover rounded cursor-zoom-in bg-gray-100"
                    />
                    <span className="text-xs text-gray-500 max-w-[160px] truncate" title={fmtName(m)}>{fmtName(m)}</span>
                    <span className="text-[11px] text-gray-400">{fmtDate(m)} · {fmtGender(m)}</span>
                    {m._distance != null && (
                      <span className="text-[11px] text-gray-400 tabular-nums">{t('common.distance')} {m._distance.toFixed(2)}</span>
                    )}
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
                    className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 transition-colors"
                  >
                    <ArrowLeft size={15} />
                  </button>
                  <span className="text-xs text-gray-400 tabular-nums w-12 text-center">{matchIndex + 1} / {total}</span>
                  <button
                    type="button"
                    onClick={() => setMatchIndex(i => Math.min(total - 1, i + 1))}
                    disabled={matchIndex === total - 1}
                    className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30 transition-colors"
                  >
                    <ArrowRight size={15} />
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}

export default {
  label: 'mode.identity',
  tasks: ['faces'],
  model: 'identity',
  initialState,
  Inputs,
  Result,
  submit,
  canSubmit: ({ images }) => images.length > 0,
  multi: true,
};
