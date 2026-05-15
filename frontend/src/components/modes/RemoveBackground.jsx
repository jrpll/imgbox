import { apiPost } from '../../lib/api';

async function submit({ image, state }) {
  const fd = new FormData();
  fd.append('image', image);
  const r = await apiPost('/remove-background', fd);
  return { blob: await r.blob(), state };
}

export default {
  label: 'remove background',
  initialState: {},
  Inputs: () => null,
  submit,
  canSubmit: ({ image }) => !!image,
};
