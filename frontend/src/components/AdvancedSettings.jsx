import { useState } from 'react';
import { CaretDown } from '@phosphor-icons/react';
import { useLang } from '../lib/i18n';

export default function AdvancedSettings({ children }) {
  const { t } = useLang();
  // Stays the way it was left, across modes and reloads.
  const [open, setOpen] = useState(() => localStorage.getItem('imgbox:advanced') === 'true');
  const toggle = () => {
    localStorage.setItem('imgbox:advanced', String(!open));
    setOpen(!open);
  };
  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        onClick={toggle}
        className="flex items-center justify-between w-full text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
      >
        {t('common.advanced')}
        <CaretDown size={12} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && children}
    </div>
  );
}
