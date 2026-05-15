![imgbox logo](frontend/src/assets/imgbox_2.png)

## Testing

Start backend

```bash
cd server && uv run uvicorn app:app --reload --port 8080
```

Start the UI

```bash
cd frontend && npm run dev
```

## Todo

- [ ] Ajouter clear cache option dans le panel de droite
- [ ] Renommer les items dans le drop down "edit" -> "sd3 + gommette edit", ainsi que les routes
- [ ] Mettre des pastilles i à côté des champs pour donner des indications

## Valider les changements

```bash
git add .
git commit -m ""
git push origin main
```