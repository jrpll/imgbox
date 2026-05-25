import { User, X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';

const initialState = { caption: '' };

function Inputs({ state, setState }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600">{t('common.caption')}</span>
        <div className="relative">
          <textarea
            value={state.caption}
            onChange={(e) => set({ caption: e.target.value })}
            rows={3}
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

async function submit({ images, state }) {
  const fd = new FormData();
  for (const img of images) fd.append('images', img);
  if (state?.caption) fd.append('caption', state.caption);
  const r = await apiPost('/identity', fd);
  const data = await r.json();
  return { state, meta: data };
}

function Result({ meta, onZoom }) {
  const { t } = useLang();
  if (!meta) {
    return (
      <div className="flex flex-col items-center gap-2 text-gray-300">
        <User size={40} />
      </div>
    );
  }
  const { processed, faces_found, skipped, ids } = meta;
  return (
    <div className="w-full h-full flex flex-col gap-3 overflow-hidden">
      <div className="text-sm text-gray-600 flex-shrink-0">
        {t('identity.summary', { processed, faces_found, skipped })}
      </div>
      <div className="flex-1 overflow-y-auto grid grid-cols-4 gap-2 auto-rows-max">
        {ids.map((id) => {
          const url = `/identity/crop/${id}`;
          return (
            <img
              key={id}
              src={url}
              onClick={() => onZoom(url)}
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
              className="w-full aspect-square object-cover rounded cursor-zoom-in bg-gray-100"
            />
          );
        })}
      </div>
    </div>
  );
}

export default {
  label: 'mode.identity',
  maxImages: 'unlimited',
  initialState,
  Inputs,
  Result,
  submit,
  canSubmit: ({ images }) => images.length > 0,
};
