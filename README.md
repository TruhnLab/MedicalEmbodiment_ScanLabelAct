# Project page: MedicalEmbodiment_ScanLabelAct

The source of the GitHub Pages site of https://github.com/TruhnLab/MedicalEmbodiment_ScanLabelAct. This folder is
its own git repository with a single branch, `gh-pages`, pushed to that repository (`git push` sends only
`gh-pages`). The code lives on `main`, in the separate `LanderPi` folder.

Plain static files, no build step at serving time; `.nojekyll` tells GitHub Pages to serve the files as they are.
The page loads nothing from other sites: the font, three.js and all media are in `assets/`.

## Preview

The page uses JavaScript modules and loads data with `fetch`, so it needs a local web server (opening `index.html`
as a file does not work). `tools/serve.py` is one with HTTP range requests, which the browsers need to seek in the
videos (the teaser's chapter buttons); `python -m http.server` has none.

```
python tools/serve.py
```

then open http://127.0.0.1:8800/.

## Layout

```
index.html            the page
assets/css/site.css   styles
assets/js/site.js     reveal on scroll, teaser chapters, exploration runs, demo tabs
assets/js/room3d.js   the labelled 3D room (three.js)
assets/data/          room.bin + room.json (3D room), runs.json + runs/*.png (exploration runs)
assets/media/         videos, posters and pictures
assets/vendor/three/  three.js r180 (MIT, see its LICENSE)
assets/fonts/         Inter (SIL Open Font License, see OFL.txt)
tools/                scripts that make assets/data and assets/media from the code repository's data
```

## Making the assets

The scripts read `../LanderPi/data/` (recordings, not in git) and use the `landerpi` conda environment.

```
conda run -n landerpi python tools/export_room.py       # assets/data/room.bin, room.json (the kitchen)
conda run -n landerpi python tools/export_runs.py       # assets/data/runs.json, runs/*.png (best run per room)
node tools/capture_orbit.mjs <frames>                   # the 3D room turning, from tools/capture.html
conda run -n landerpi python tools/make_media.py --frames <frames>   # videos, posters, pictures
```

`capture_orbit.mjs` needs `puppeteer-core`, the local web server above (`tools/serve.py`) and a headless Edge or
Chrome with remote debugging on port 9333 (see the script's header).
