// Backfill Youtarr .nfo + poster .jpg for imported videos (files that have no .nfo next to them).
const fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');
const nfo = require('/app/server/modules/nfoGenerator');
const DATA = process.env.DATA_PATH || '/usr/src/app/data', INFO = '/app/jobs/info';
let done = 0, skipped = 0, noinfo = 0, noimg = 0;
function walk(d) { for (const e of fs.readdirSync(d, { withFileTypes: true })) {
  const p = path.join(d, e.name);
  if (e.isDirectory()) { if (!e.name.startsWith('.') && e.name !== '__playlists__') walk(p); continue; }
  const m = e.name.match(/\[([A-Za-z0-9_-]{11})\]\.(mp4|mkv|webm)$/); if (!m) continue;
  const base = p.replace(/\.[^.]+$/, '');
  if (fs.existsSync(base + '.nfo')) { skipped++; continue; }
  const ij = path.join(INFO, m[1] + '.info.json');
  if (!fs.existsSync(ij)) { noinfo++; continue; }
  const j = JSON.parse(fs.readFileSync(ij, 'utf8'));
  nfo.writeVideoNfoFile(p, j);
  if (!fs.existsSync(base + '.jpg')) {
    const url = j.thumbnail || `https://i.ytimg.com/vi/${m[1]}/maxresdefault.jpg`;
    try { execFileSync('curl', ['-sfL', '-m', '30', '-o', base + '.jpg', url]); } catch { noimg++; }
  }
  done++;
} }
walk(DATA);
console.log(JSON.stringify({ written: done, alreadyHadNfo: skipped, missingInfoJson: noinfo, thumbFailures: noimg }));
