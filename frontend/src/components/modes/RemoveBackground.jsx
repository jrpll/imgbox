import { apiPost } from '../../lib/api';

async function submit({ image }) {
  const fd = new FormData();
  fd.append('image', image);
  const r = await apiPost('/remove-background', fd);
  return URL.createObjectURL(await r.blob());
}

export default {
  label: 'remove background',
  initialState: {},
  Inputs: () => null,
  submit,
  canSubmit: ({ image }) => !!image,
};
