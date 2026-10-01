from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CASES = ('rings_closed_reference', 'rings_closed_graph', 'rings_matched_A', 'rings_matched_B')
VMAX = 1.25


def pack(values):
    source = np.asarray(values, dtype=np.uint8).ravel()
    output = bytearray()
    i = 0
    while i < len(source):
        run = 1
        while i + run < len(source) and source[i + run] == source[i] and run < 255:
            run += 1
        if run >= 3:
            output.extend((0, run, int(source[i])))
            i += run
        else:
            start = i
            i += run
            while i < len(source) and i - start < 255:
                if i + 2 < len(source) and source[i] == source[i + 1] == source[i + 2]:
                    break
                i += 1
            output.append(i - start)
            output.extend(source[start:i].tobytes())
    return base64.b64encode(output).decode('ascii')


def unpack(encoded, length):
    source = base64.b64decode(encoded)
    output = bytearray()
    i = 0
    while i < len(source):
        count = source[i]
        i += 1
        if count:
            output.extend(source[i:i + count])
            i += count
        else:
            output.extend([source[i + 1]] * source[i])
            i += 2
    assert len(output) == length
    return np.frombuffer(output, dtype=np.uint8)


def contours(field, level, centres=True):
    n = field.shape[0]
    coordinate = (np.arange(n) + (.5 if centres else 0)) * 60 / n
    fig, ax = plt.subplots()
    contour = ax.contour(coordinate, coordinate, field.T, levels=[level])
    segments = [segment.round(5).tolist() for segment in contour.allsegs[0] if len(segment) > 1]
    plt.close(fig)
    return segments


def resolve_source(source: Path) -> Path:
    source = Path(source)
    candidates = [source]
    if source.name == 'source':
        candidates.append(source.parent)
    elif source.name == '.research':
        candidates.append(source / 'source')
    for candidate in candidates:
        if all((candidate / 'media_data' / (name + '.npz')).is_file() for name in CASES):
            return candidate
    raise FileNotFoundError(
        'Dense ring states are missing. Run python code/reproduce.py --media first, '
        'or set --source to a research directory containing all four media_data/rings_*.npz files. '
        'The supplied directory takes priority; a legacy .research/source path may fall back to .research.'
    )


def build(source: Path, output: Path):
    source = resolve_source(source)
    if output.suffix.lower() != '.html':
        output = output / 'rings.html'
    samples = np.arange(0, 211, 2)
    cases = []
    provenance = []
    for name in CASES:
        path = source / 'media_data' / (name + '.npz')
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            raw = z['voltage'][samples]
            quantized = np.rint(np.clip(raw / VMAX, 0, 1) * 255).astype(np.uint8)
            encoded = pack(quantized)
            np.testing.assert_array_equal(unpack(encoded, quantized.size), quantized.ravel())
            error = float(np.max(abs(raw - quantized.astype(float) * VMAX / 255)))
            assert error <= VMAX / 510 + 1e-7
            target = z['target_mask']
            activation = z['activation'][target]
            events = np.sort(activation[np.isfinite(activation)]).tolist()
            score = z['source_score']
            substrate = contours(score, 0, centres=False) if score.size else contours(z['diffusivity'], .25)
            cases.append(dict(
                id=name, title=meta['title'], group=meta['group'], frames=encoded,
                substrate=substrate, target=contours(target.astype(float), .5),
                events=events, targetCells=int(target.sum()), capacity=meta['normalized_capacity'],
                arrival=meta['outcome']['first_arrival_ms'], captured=meta['outcome']['captured_by_210_ms'],
                width=meta.get('width_mm'), eta=meta.get('nominal_gap_diffusivity'), angle=meta['angle_deg'],
            ))
            provenance.append(dict(
                id=name, source_npz='media_data/' + path.name,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                numerical_provenance=meta, target_cells=int(target.sum()),
                target_activated_fraction=float(np.isfinite(activation).mean()),
                rendered_frames=len(samples), sampled_times_ms=samples.tolist(),
                source_voltage_storage=str(raw.dtype), render_voltage_storage='uint8',
                render_voltage_range=[0, VMAX], maximum_render_quantization_error=error,
                quantization_only_for_colour_rendering=True, activation_metrics_full_precision=True,
                temporal_interpolation=False, spatial_display='Canvas smoothing of the fixed finite-volume grid',
            ))
    lut = np.rint(255 * plt.colormaps['magma'](np.linspace(0, 1, 256))[:, :3]).astype(int).tolist()
    payload = dict(cases=cases, palette=lut, times=samples.tolist(), n=121, maxVoltage=VMAX)
    page = TEMPLATE.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'), allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    report = dict(
        source='Saved original-model dense reruns for manuscript ring experiments',
        spatial_domain_mm=[60, 60], production_grid=[121, 121], solver_dt_ms=.02,
        pacing_direction='exit', physical_horizon_ms=210, frame_sample_interval_ms=2,
        codec='PackBits-style RLE embedded as base64; no external requests',
        cases=provenance, output_bytes=output.stat().st_size,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    output.with_name('rings_provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'html': str(output), 'bytes': output.stat().st_size, 'frames_per_case': len(samples)}, indent=2))


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Explore computed atrial barrier ring experiments: synchronized monodomain voltage and distal activation on identical numerical protocols.">
<title>Ring propagation explorer</title>
<style>
:root{color-scheme:dark;--bg:#07111d;--panel:#102131;--ink:#edf5fb;--muted:#a5b8c9;--line:#294152;--cyan:#67dfdd;--orange:#ffb26b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select,input{font:inherit}button,select{color:var(--ink);background:#162c3f;border:1px solid var(--line);border-radius:9px;padding:9px 13px}button{cursor:pointer}button:hover{border-color:#7dbbc7;background:#1b354a}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid var(--cyan);outline-offset:4px}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}main{max-width:1240px;margin:auto;padding:24px 28px 28px}.topline{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:20px}.back{font-size:13px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}h1{font-size:clamp(23px,3.6vw,37px);line-height:1.2;letter-spacing:-.025em;margin:10px 0 8px;font-weight:650}.subtitle{color:var(--muted);margin:0;max-width:800px}.scenario{display:flex;gap:8px;margin:23px 0 18px;flex-wrap:wrap}.scenario button{font-size:14px;padding:10px 17px}.scenario button[aria-pressed=true]{color:var(--bg);background:var(--cyan);border-color:var(--cyan);font-weight:650}.controls{display:flex;align-items:center;gap:13px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:13px;flex-wrap:wrap}.play{min-width:94px;background:#20394a;font-weight:600}.clock{font:600 23px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;min-width:104px;color:var(--ink);font-variant-numeric:tabular-nums}.scrub{flex:1;min-width:130px;accent-color:var(--cyan);height:26px}.timehint{font-size:11px;color:var(--muted);display:block;margin-top:4px}.speed{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px}.speed select{padding:6px 8px;font-size:13px}.row{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin:13px 2px 14px}.toggles{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted)}.toggles label{cursor:pointer;display:flex;align-items:center;gap:6px}.toggles input{accent-color:var(--cyan)}.protocol{font-size:12px;color:var(--muted)}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}.case{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}.casehead{padding:16px 18px 11px;display:flex;justify-content:space-between;align-items:flex-start;gap:10px}.case h2{margin:0 0 3px;font-size:18px;font-weight:640;line-height:1.2}.case.a h2,.case.a .live{color:var(--cyan)}.case.b h2,.case.b .live{color:var(--orange)}.caseinfo{font-size:12px;color:var(--muted);min-height:19px}.live{text-align:right;font-size:23px;font-weight:620;line-height:1.1;font-variant-numeric:tabular-nums}.live small{display:block;font-size:10px;letter-spacing:.02em;color:var(--muted);font-weight:450;margin-top:5px}.field{display:block;width:100%;height:auto;aspect-ratio:1;background:#000}.casefoot{padding:11px 17px 13px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.casefoot strong{font-weight:580;color:var(--ink)}.legend{margin:16px auto 20px;max-width:570px;display:flex;gap:13px;align-items:center;color:var(--muted);font-size:12px}.gradientwrap{flex:1}.gradient{height:11px;border-radius:4px;display:block;width:100%}.ticks{display:flex;justify-content:space-between;font-size:10px;margin-top:3px}.tracebox{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px 18px 10px}.tracehead{display:flex;gap:10px;justify-content:space-between;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted)}.tracehead strong{font-size:14px;color:var(--ink);font-weight:600}.series{display:flex;gap:16px;font-size:11px}.series span:before{content:"";display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:6px;background:var(--cyan)}.series span:last-child:before{background:var(--orange)}#trace{display:block;width:100%;height:170px}.findings{margin:18px 2px 14px;color:var(--muted);font-size:13px;line-height:1.7}.findings strong{color:var(--ink);font-weight:550}details{border-top:1px solid var(--line);padding-top:13px;color:var(--muted);font-size:12px}summary{cursor:pointer;color:var(--ink);font-size:12px}details p{max-width:960px}footer{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;margin-top:16px;color:#829aad;font-size:11px}.error{padding:15px;color:#ffb26b;border:1px solid #ffb26b;display:none}@media(max-width:700px){main{padding:16px 14px}.plots{gap:12px}.casehead{padding:12px;flex-direction:column}.live{font-size:19px;text-align:left}.live small{display:inline;margin-left:6px}.case h2{font-size:15px}.caseinfo{font-size:10px}.casefoot{font-size:10px;padding:9px 12px}.controls{gap:9px;padding:11px}.clock{font-size:19px;min-width:80px}.speed{order:4}.row{gap:10px}.protocol{font-size:11px}.legend{font-size:11px}#trace{height:155px}.tracebox{padding:12px 9px 8px}}@media(max-width:430px){.plots{grid-template-columns:1fr}.casehead{flex-direction:row}.case h2{font-size:17px}.caseinfo{font-size:12px}.casefoot{font-size:12px}.field{max-height:none}.live small{display:block;margin-left:0}.live{text-align:right}.topline{margin-bottom:12px}.scenario button{font-size:12px;padding:9px 11px}.scrub{min-width:115px}.protocol{width:100%}}
</style>
</head>
<body><main>
<div class="topline"><a class="back" href="../index.html">← Research overview</a><span class="eyebrow">Interactive simulations</span><button id="fullscreen" type="button" aria-label="Enter fullscreen" title="Fullscreen">⛶</button></div>
<h1 id="heading">Closed contours can still conduct</h1>
<p class="subtitle" id="subtitle">The same exit-pacing protocol applied to a reference ring and its continuous graph reconstruction.</p>
<div class="scenario" role="group" aria-label="Simulation comparison"><button type="button" data-scenario="closed" aria-pressed="true">Closed contour</button><button type="button" data-scenario="matched" aria-pressed="false">Matched conductance</button></div>
<div class="controls"><button class="play" id="play" type="button" aria-label="Play simulation">▶ Play</button><div><output class="clock" id="clock" aria-live="off">0 ms</output><span class="timehint">Physical time</span></div><input class="scrub" id="scrub" type="range" min="0" max="210" value="0" step="2" aria-label="Physical simulation time in milliseconds"><label class="speed">Playback<select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label></div>
<div class="row"><div class="toggles"><label><input type="checkbox" id="substrate" checked>Substrate outline</label><label><input type="checkbox" id="target" checked>Distal sector</label><label><input type="checkbox" id="loop" checked>Loop</label></div><span class="protocol">Exit pacing · 121 × 121 cells · Δt = 0.02 ms</span></div>
<div class="error" id="error" role="alert"></div>
<div class="plots"><article class="case a"><div class="casehead"><div><h2 id="title0"></h2><div class="caseinfo" id="info0"></div></div><div class="live"><span id="fraction0">0%</span><small>target activated</small></div></div><canvas class="field" id="field0" width="600" height="600" role="img" aria-label="First case voltage field"></canvas><div class="casefoot"><span id="capacity0"></span><span id="arrival0"></span></div></article><article class="case b"><div class="casehead"><div><h2 id="title1"></h2><div class="caseinfo" id="info1"></div></div><div class="live"><span id="fraction1">0%</span><small>target activated</small></div></div><canvas class="field" id="field1" width="600" height="600" role="img" aria-label="Second case voltage field"></canvas><div class="casefoot"><span id="capacity1"></span><span id="arrival1"></span></div></article></div>
<div class="legend"><span>Normalized voltage V</span><div class="gradientwrap"><canvas id="gradient" class="gradient" width="256" height="10" aria-hidden="true"></canvas><div class="ticks"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span><span>1.25</span></div></div></div>
<div class="tracebox"><div class="tracehead"><strong>Distal-sector activation</strong><div class="series"><span id="series0"></span><span id="series1"></span></div></div><canvas id="trace" role="img" aria-label="Cumulative target activation versus physical time"></canvas></div>
<p class="findings" id="finding"></p>
<details><summary>Protocol and numerical data</summary><p>These are four precomputed monodomain simulations on a 60 × 60 mm sheet, with no-flux boundaries and longitudinal/transverse diffusivities 0.32/0.0512 mm²/ms. Exit pacing uses the original central stimulus: radius 3.5 mm, duration 2 ms and amplitude 1.2 normalized voltage/ms. Capture requires activation of at least 80% of the marked distal sector by 210 ms. Reported first arrival is conditional on this final capture outcome.</p><p>The underlying solver step is 0.02 ms. This viewer displays actual states every 2 ms without temporal interpolation; colour values alone are quantized to 8 bits on the common range 0–1.25. Target activation times and conductance values retain their saved precision. The white outlines show the zero-score contour in the reconstruction comparison and relative diffusivity η = 0.25 in the matched comparison. <a href="rings_provenance.json">Numerical provenance</a>.</p><p>These controlled synthetic results compare barrier geometry, passive conductance and finite-horizon propagation. They do not establish a clinical isolation test or a universal conductance threshold. The exit-directed matched-case contrast persists over the spatial and temporal refinements reported in the manuscript.</p></details>
<footer><span>Computed fields · shared colour scale · 210 ms horizon</span><span>Space: play/pause · ← / →: one saved frame</span></footer>
</main>
<script id="simulation-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('simulation-data').textContent), $=id=>document.getElementById(id);
const colours=['#67dfdd','#ffb26b'], contexts=[$('field0').getContext('2d'),$('field1').getContext('2d')];
const buffers=[document.createElement('canvas'),document.createElement('canvas')];
buffers.forEach(c=>{c.width=data.n;c.height=data.n;});
const bufferContexts=buffers.map(c=>c.getContext('2d'));
let scenario='closed', chosen=[], frame=0, playing=false, previous=0, elapsed=0, animationRequest=0;
function decode(text,size){const input=Uint8Array.from(atob(text),c=>c.charCodeAt(0)),output=new Uint8Array(size);let i=0,j=0;while(i<input.length){const count=input[i++];if(count){output.set(input.subarray(i,i+count),j);i+=count;j+=count;}else{const run=input[i++],value=input[i++];output.fill(value,j,j+run);j+=run;}}if(j!==size)throw Error('Frame data length mismatch');return output;}
function pixelsFor(c){if(!c.pixels)c.pixels=decode(c.frames,data.times.length*data.n*data.n);return c.pixels;}
function fraction(c,t){let low=0,high=c.events.length;while(low<high){const middle=(low+high)>>1;if(c.events[middle]<=t)low=middle+1;else high=middle;}return low/c.targetCells;}
function strokePaths(ctx,paths,colour,width){ctx.strokeStyle=colour;ctx.lineWidth=width;ctx.beginPath();for(const path of paths){path.forEach((p,i)=>{const x=p[0]*10,y=600-p[1]*10;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});}ctx.stroke();}
function field(i){const c=chosen[i],ctx=contexts[i],n=data.n,bytes=pixelsFor(c),offset=frame*n*n,image=bufferContexts[i].createImageData(n,n);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const palette=data.palette[bytes[offset+x*n+(n-1-y)]],k=4*(y*n+x);image.data[k]=palette[0];image.data[k+1]=palette[1];image.data[k+2]=palette[2];image.data[k+3]=255;}bufferContexts[i].putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(buffers[i],0,0,600,600);if($('substrate').checked)strokePaths(ctx,c.substrate,'rgba(237,245,251,.75)',1.4);if($('target').checked)strokePaths(ctx,c.target,colours[i],2.3);ctx.strokeStyle='rgba(165,184,201,.6)';ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.beginPath();ctx.arc(300,300,35,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle='#a5b8c9';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(44,45);ctx.lineTo(137,45);ctx.lineTo(131,41);ctx.moveTo(137,45);ctx.lineTo(131,49);ctx.stroke();ctx.font='13px system-ui';ctx.fillStyle='#a5b8c9';ctx.fillText('fibres',44,66);ctx.strokeStyle='#edf5fb';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(44,555);ctx.lineTo(144,555);ctx.stroke();ctx.font='13px system-ui';ctx.fillStyle='#edf5fb';ctx.fillText('10 mm',70,545);$('fraction'+i).textContent=Math.round(100*fraction(c,data.times[frame]))+'%';$('field'+i).setAttribute('aria-label',c.title+'; normalized voltage at '+data.times[frame]+' milliseconds; '+Math.round(100*fraction(c,data.times[frame]))+' percent of target activated');}
function trace(){const canvas=$('trace'),rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(200,rect.width),h=rect.height;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const left=40,right=w-27,top=17,bottom=h-33,x=t=>left+(right-left)*t/210,y=f=>bottom-(bottom-top)*f,now=data.times[frame];ctx.clearRect(0,0,w,h);ctx.font='11px system-ui';ctx.strokeStyle='#294152';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();ctx.fillStyle='#a5b8c9';ctx.textAlign='right';for(const val of [0,.8,1])ctx.fillText((100*val)+'%',left-8,y(val)+4);ctx.setLineDash([4,5]);ctx.strokeStyle='rgba(165,184,201,.45)';ctx.beginPath();ctx.moveTo(left,y(.8));ctx.lineTo(right,y(.8));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#a5b8c9';ctx.font='10px system-ui';ctx.textAlign='right';ctx.fillText('capture criterion',right,y(.8)-6);ctx.textAlign='center';for(const t of [0,50,100,150,210])ctx.fillText(t,x(t),bottom+17);ctx.fillText('Physical time (ms)',(left+right)/2,h-2);function line(c,index,until,alpha){ctx.beginPath();ctx.moveTo(x(0),y(0));let count=0;for(const t of c.events){if(t>until)break;ctx.lineTo(x(t),y(count/c.targetCells));count++;ctx.lineTo(x(t),y(count/c.targetCells));}ctx.lineTo(x(until),y(count/c.targetCells));ctx.strokeStyle=colours[index];ctx.globalAlpha=alpha;ctx.lineWidth=alpha===1?2:1;ctx.setLineDash(index===0?[5,3]:[]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;}chosen.forEach((c,i)=>{line(c,i,210,.22);line(c,i,now,1);ctx.fillStyle=colours[i];ctx.beginPath();ctx.arc(x(now),y(fraction(c,now)),3.1,0,Math.PI*2);ctx.fill();});ctx.strokeStyle='rgba(237,245,251,.45)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(now),top);ctx.lineTo(x(now),bottom);ctx.stroke();}
function render(){const t=data.times[frame];$('clock').textContent=t+' ms';$('scrub').value=t;field(0);field(1);trace();}
function choose(name){scenario=name;chosen=data.cases.filter(c=>c.group===name);document.querySelectorAll('[data-scenario]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scenario===name)));$('heading').textContent=name==='closed'?'Closed contours can still conduct':'Nearly equal conductance, different propagation';$('subtitle').textContent=name==='closed'?'The same exit-pacing protocol applied to a reference ring and its continuous graph reconstruction.':'Two geometrically and materially different gaps, selected by passive conductance before examining propagation.';chosen.forEach((c,i)=>{$('title'+i).textContent=c.title;$('series'+i).textContent=c.title;$('info'+i).textContent=name==='closed'?(i?'Zero thresholded gaps · 2 mm blackout':'Zero thresholded gaps'):c.width+' mm gap · η = '+c.eta+' · '+c.angle+'°';$('capacity'+i).innerHTML='C* = <strong>'+c.capacity.toFixed(7)+'</strong>';$('arrival'+i).innerHTML=c.arrival===null?'No target capture by 210 ms':'First arrival <strong>'+c.arrival.toFixed(2)+' ms</strong>';});$('finding').innerHTML=name==='closed'?'<strong>Both thresholded contours are closed.</strong> The reference blocks exit capture over the 210 ms horizon; the reconstructed continuous substrate transmits excitation. A closed score contour does not determine electrical block.':'<strong>Passive conductance differs by only 0.098%.</strong> Case A captures the distal sector; case B does not under the same exit-pacing protocol. Similar passive conductance accompanies different nonlinear propagation in these computed cases.';render();}
function play(state){cancelAnimationFrame(animationRequest);playing=state;$('play').textContent=state?'❚❚ Pause':'▶ Play';$('play').setAttribute('aria-label',state?'Pause simulation':'Play simulation');if(state){if(frame===data.times.length-1){frame=0;elapsed=0;render();}previous=performance.now();animationRequest=requestAnimationFrame(tick);}}
function tick(timestamp){if(!playing)return;elapsed+=(timestamp-previous)*.015*Number($('speed').value);previous=timestamp;if(elapsed>=210){if($('loop').checked){if(elapsed>=230)elapsed%=230;}else{elapsed=210;play(false);}}const next=Math.min(105,Math.floor(elapsed/2));if(next!==frame){frame=next;render();}if(playing)animationRequest=requestAnimationFrame(tick);}
$('play').onclick=()=>play(!playing);$('scrub').oninput=()=>{elapsed=Number($('scrub').value);frame=Math.round(elapsed/2);render();};['substrate','target'].forEach(id=>$(id).onchange=render);document.querySelectorAll('[data-scenario]').forEach(b=>b.onclick=()=>choose(b.dataset.scenario));$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{window.open(location.href,'_blank','noopener');}};window.addEventListener('resize',trace);document.addEventListener('keydown',event=>{if(['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(event.code==='Space'){event.preventDefault();play(!playing);}else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();play(false);frame=Math.max(0,Math.min(105,frame+(event.key==='ArrowRight'?1:-1)));elapsed=data.times[frame];render();}});
const gradient=$('gradient').getContext('2d');data.palette.forEach((p,i)=>{gradient.fillStyle='rgb('+p.join(',')+')';gradient.fillRect(i,0,1,10);});
const requestedCase=new URLSearchParams(window.location.search).get('case');
try{choose(requestedCase==='matched'?'matched':'closed');}catch(error){$('error').style.display='block';$('error').textContent='The numerical frames could not be decoded: '+error.message;}
</script>
</body></html>'''


if __name__ == '__main__':
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Build the offline ring explorer from saved numerical states.',
        epilog='Run python code/reproduce.py --media first if dense ring states are missing.'
    )
    parser.add_argument('--source', type=Path, default=repository / '.research',
                        help='Research directory containing media_data/rings_*.npz (default: repository .research). The supplied directory takes priority; legacy .research/source paths fall back to .research.')
    parser.add_argument('--output', type=Path, default=repository / 'docs/interactive',
                        help='Output directory or HTML path (default: repository docs/interactive).')
    args = parser.parse_args()
    build(args.source, args.output)
