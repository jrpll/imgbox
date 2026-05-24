import { User } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';

async function submit({ images }) {
  const fd = new FormData();
  for (const img of images) fd.append('images', img);
  const r = await apiPost('/identity', fd);
  const data = await r.json();
  return { state: {}, meta: data };
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
  initialState: {},
  Inputs: () => null,
  Result,
  submit,
  canSubmit: ({ images }) => images.length > 0,
};
