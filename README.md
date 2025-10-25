## Installation

Run git clone.

Install python packages for backend :

```bash
pip install -r imgbox/server/requirements.txt
```
Then install the node packages for the frontend :

```bash
cd client
npm install
```

## Hosting on A100 virtual machine

Prepare for production :

```bash
npm run build
```

Launch the server : 

```bash
cd server
nohup /gunicorn/parent/folder/gunicorn --timeout 600 -w 1 -b 0.0.0.0:8080 app:app
```

## Installing and running on 24GB VRAM (ex : RTX4090)

Run :

```bash 
python server/app.py
```

Then, run :

```bash
cd client
npm run dev
```

## TODO :

- [ ] Add automatic face crop extraction
- [ ] Add background remover
- [ ] Add Kontext and PulID post processors
- [ ] Revamp interface / unslop
