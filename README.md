<div align="center">
  <img src="client/dist/box.svg" alt="imgbox logo" width="100"/>
  <h1>imgbox</h1>
</div>

## Installation

Download code : 

```bash
git clone https://github.com/jrpll/imgbox.git 
```

Install python packages for backend :

```bash
pip install -r imgbox/server/requirements.txt
```
Then install the node packages for the frontend :

```bash
cd client
npm install
```

Get a huggingface token, which you will put under .env file in server.

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
