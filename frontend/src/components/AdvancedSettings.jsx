import { useState } from 'react';
import { CaretLeft } from '@phosphor-icons/react';
import { useLang } from '../lib/i18n';

export default function AdvancedSettings({ children }) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex items-center justify-between w-full text-xs text-gray-400 hover:text-gray-600 transition-colors"
      >
        {t('common.advanced')}
        <CaretLeft size={12} className={`transition-transform ${open ? '-rotate-90' : ''}`} />
      </button>
      {open && children}
    </div>
  );
}
