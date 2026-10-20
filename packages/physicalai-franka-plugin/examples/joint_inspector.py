"""Read-only joint inspector: live FR3 and GELLO values side by side in a browser.

Nothing is ever commanded. The robots are only read with ``Robot.read_once`` (FCI must be
active), the hands with ``Gripper.read_once``, and the GELLO servos are only read (torque off).
"Stop reading" closes every connection, so other programs can use the robots again.

Run on the robot PC, for example::

    python examples/joint_inspector.py \
        --robot left=172.16.1.49 --robot right=172.16.0.41 \
        --gello BE9F=/dev/serial/by-id/usb-ROBOTIS_..BE9F..-if00,~/gello-calibration/right.json \
        --gello 1CA8=/dev/serial/by-id/usb-ROBOTIS_..1CA8..-if00,~/gello-calibration/left.json \
        --pair left=BE9F --pair right=1CA8

Then on your laptop: ``ssh -L 8765:localhost:8765 user@robot-pc`` and open http://localhost:8765.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from gello_teleop import JOINT_LIMIT_CHOICES, joint_limits_deg, use_gello_limits  # same folder

ROBOT_PERIOD_S = 0.05
GELLO_PERIOD_S = 0.03
GRIPPER_PERIOD_S = 0.5
RETRY_S = 2.0


def _rounded(values: Any, digits: int = 2) -> list[float]:  # noqa: ANN401
    return [round(float(value), digits) for value in values]


class _Reporter:
    """Child-process side: sends ``(key, values)`` updates to the parent's state."""

    def __init__(self, key: str, updates: Any, stop: Any) -> None:  # noqa: ANN401 - multiprocessing objects
        self.key = key
        self._updates = updates
        self.stop = stop

    def set(self, **values: Any) -> None:  # noqa: ANN401
        self._updates.put((self.key, {**values, "updated": time.time()}))

    def retry_wait(self) -> None:
        self.stop.wait(RETRY_S)


def _robot_loop(report: _Reporter, host: str) -> None:
    while not report.stop.is_set():
        robot = None
        try:
            report.set(host=host, status="connecting", error=None)
            import pylibfranka  # noqa: PLC0415

            robot = pylibfranka.Robot(host, pylibfranka.RealtimeConfig.kIgnore)
            report.set(status="reading")
            while not report.stop.is_set():
                state = robot.read_once()
                o_t_ee = list(state.O_T_EE)
                report.set(
                    q_deg=_rounded(np.degrees(state.q)),
                    q_d_deg=_rounded(np.degrees(state.q_d)),
                    dq_deg_s=_rounded(np.degrees(state.dq)),
                    tau_ext_nm=_rounded(state.tau_ext_hat_filtered),
                    joint_contact=_rounded(state.joint_contact, 0),
                    ee_xyz_m=_rounded(o_t_ee[12:15], 4),
                    robot_mode=str(state.robot_mode).rsplit(".", 1)[-1],
                    current_errors=str(state.current_errors)[:300],
                )
                report.stop.wait(ROBOT_PERIOD_S)
        except Exception as exc:  # noqa: BLE001  # keep retrying, show the reason on the page
            report.set(status="error", error=f"{type(exc).__name__}: {exc}"[:400])
            report.retry_wait()
        finally:
            del robot
    report.set(status="stopped")


def _gripper_loop(report: _Reporter, host: str) -> None:
    while not report.stop.is_set():
        gripper = None
        try:
            import pylibfranka  # noqa: PLC0415

            gripper = pylibfranka.Gripper(host)
            while not report.stop.is_set():
                state = gripper.read_once()
                report.set(
                    status="reading",
                    error=None,
                    width_mm=round(state.width * 1000, 1),
                    max_width_mm=round(state.max_width * 1000, 1),
                    is_grasped=bool(state.is_grasped),
                )
                report.stop.wait(GRIPPER_PERIOD_S)
        except Exception as exc:  # noqa: BLE001
            report.set(status="error", error=f"{type(exc).__name__}: {exc}"[:400])
            report.retry_wait()
        finally:
            del gripper
    report.set(status="stopped")


def _gello_loop(report: _Reporter, port: str, calibration_path: str | None, limits_deg: list[float]) -> None:
    from physicalai_franka_plugin.gello import _RAD_PER_COUNT, GelloLeader  # noqa: PLC0415

    use_gello_limits(np.asarray(limits_deg))  # a spawned child does not inherit the parent's patch
    calibration = Path(calibration_path).expanduser().read_text() if calibration_path else None
    while not report.stop.is_set():
        leader = GelloLeader(port, calibration=calibration)
        try:
            report.set(port=port, calibration=calibration_path, status="connecting", error=None)
            leader.connect()
            cal = leader._calibration  # noqa: SLF001
            offsets = np.asarray(cal.assembly_offsets_rad)
            signs = np.asarray(cal.joint_signs)
            report.set(
                status="reading",
                signs=list(cal.joint_signs),
                offsets_deg=_rounded(np.degrees(offsets)),
                gripper_range_deg=_rounded(np.degrees(cal.gripper_range_rad)),
            )
            while not report.stop.is_set():
                observation = leader.get_observation()
                raw = np.asarray(observation.sensor_data["raw_positions_rad"])
                unclipped = np.degrees(leader._normalize_initial(raw[:7]))  # noqa: SLF001
                report.set(
                    counts=[round(float(value / _RAD_PER_COUNT)) for value in raw],
                    raw_deg=_rounded(np.degrees(raw)),
                    offset_sign_deg=_rounded(np.degrees((raw[:7] - offsets) * signs)),
                    q_unclipped_deg=_rounded(unclipped),
                    q_deg=_rounded(observation.joint_positions[:7]),
                    trigger=round(float(observation.joint_positions[7]), 3),
                )
                report.stop.wait(GELLO_PERIOD_S)
        except Exception as exc:  # noqa: BLE001
            report.set(status="error", error=f"{type(exc).__name__}: {exc}"[:400])
            report.retry_wait()
        finally:
            leader.disconnect()
    report.set(status="stopped")


class Inspector:
    """Background readers for robots, hands and GELLOs; ``snapshot()`` returns the latest values.

    Every device is read in its own process: pylibfranka holds the GIL inside ``read_once``,
    which starved GELLO reads in a shared process past the Dynamixel packet timeout.
    """

    def __init__(
        self,
        robots: dict[str, str],
        gellos: dict[str, tuple[str, str | None]],
        limits_deg: list[float] | None = None,
    ) -> None:
        self.robots = robots
        self.gellos = gellos
        self.limits_deg = limits_deg if limits_deg is not None else joint_limits_deg("hardware").tolist()
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._stop: Any = None
        self._updates: Any = None
        self._collector: threading.Thread | None = None
        self._processes: list[Any] = []
        self.running = False

    def _set(self, key: str, **values: Any) -> None:  # noqa: ANN401
        with self._lock:
            self._state.setdefault(key, {}).update(values, updated=time.time())

    def _collect(self, updates: Any) -> None:  # noqa: ANN401
        while (item := updates.get()) is not None:
            key, values = item
            with self._lock:
                self._state.setdefault(key, {}).update(values)

    def start(self) -> None:
        if self.running:
            return
        self._state = {}
        self._stop = self._ctx.Event()
        self._updates = self._ctx.Queue()
        self._collector = threading.Thread(target=self._collect, args=(self._updates,), daemon=True)
        self._collector.start()
        jobs: list[tuple[str, Any, tuple[Any, ...]]] = []
        for name, host in self.robots.items():
            jobs.append((f"robot:{name}", _robot_loop, (host,)))
            jobs.append((f"hand:{name}", _gripper_loop, (host,)))
        for name, (port, calibration) in self.gellos.items():
            jobs.append((f"gello:{name}", _gello_loop, (port, calibration, self.limits_deg)))
        for key, target, args in jobs:
            report = _Reporter(key, self._updates, self._stop)
            process = self._ctx.Process(target=target, args=(report, *args), name=key, daemon=True)
            process.start()
            self._processes.append(process)
        self.running = True

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._processes = []
        if self._updates is not None:
            self._updates.put(None)
        if self._collector is not None:
            self._collector.join(timeout=2.0)
        self._collector = None
        self.running = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            devices = json.loads(json.dumps(self._state))
        return {"running": self.running, "time": time.time(), "devices": devices}


def make_handler(inspector: Inspector, config: dict[str, Any], snapshot_file: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ANN401
            return

        def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(json.dumps(inspector.snapshot()).encode())
            elif self.path == "/api/config":
                self._send(json.dumps(config).encode())
            else:
                self._send(b'{"error": "not found"}', status=404)

        def do_POST(self) -> None:
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if self.path == "/api/start":
                inspector.start()
            elif self.path == "/api/stop":
                inspector.stop()
            elif self.path == "/api/snapshot":
                record = {
                    "saved": datetime.now(UTC).isoformat(),
                    "note": str(body.get("note", ""))[:500],
                    "pairing": body.get("pairing"),
                    **inspector.snapshot(),
                }
                with snapshot_file.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                self._send(json.dumps({"saved_to": str(snapshot_file)}).encode())
                return
            else:
                self._send(b'{"error": "not found"}', status=404)
                return
            self._send(json.dumps({"running": inspector.running}).encode())

    return Handler


def _named(values: list[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            msg = f"{flag} expects NAME=VALUE, got {value!r}"
            raise SystemExit(msg)
        result[name] = rest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", action="append", default=[], help="NAME=HOST, e.g. left=172.16.1.49")
    parser.add_argument("--gello", action="append", default=[], help="NAME=PORT[,CALIBRATION_JSON]")
    parser.add_argument("--pair", action="append", default=[], help="ROBOT=GELLO default pairing shown on the page")
    parser.add_argument("--bind", default="127.0.0.1", help="Address to serve on (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--snapshots", default="~/joint-inspector-snapshots.jsonl")
    parser.add_argument("--joint-limits", choices=JOINT_LIMIT_CHOICES, default="hardware")
    parser.add_argument("--limit-margin-deg", type=float, default=5.0)
    args = parser.parse_args()
    active_limits = joint_limits_deg(args.joint_limits, args.limit_margin_deg)
    use_gello_limits(active_limits)  # GELLO "q used" is clipped exactly like teleop clips it

    robots = _named(args.robot, "--robot")
    gellos = {}
    for name, value in _named(args.gello, "--gello").items():
        port, _, calibration = value.partition(",")
        gellos[name] = (port, calibration or None)
    pairing = _named(args.pair, "--pair")
    config = {
        "robots": robots,
        "gellos": list(gellos),
        "pairing": pairing,
        "limits_name": f"{args.joint_limits}"
        + (f" − {args.limit_margin_deg:g}°" if args.joint_limits == "hardware" else ""),
        "limits_deg": active_limits.round(1).tolist(),
        "limit_sets": {
            "FR3 hardware": joint_limits_deg("hardware", 0.0).round(1).tolist(),
            "Franka suggested": joint_limits_deg("suggested").round(1).tolist(),
            "plugin (URDF)": joint_limits_deg("plugin").round(1).tolist(),
        },
    }
    inspector = Inspector(robots, gellos, active_limits.tolist())
    snapshot_file = Path(args.snapshots).expanduser()
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(inspector, config, snapshot_file))
    print(f"Joint inspector on http://{args.bind}:{args.port}  (read-only; Ctrl+C to quit)")
    print(f"Robots {robots}; GELLOs {list(gellos)}; snapshots -> {snapshot_file}")
    # pkill sends SIGTERM: exit through the finally below so the reader processes release their ports
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        inspector.stop()
        server.server_close()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>FR3 / GELLO joint inspector</title>
<style>
 body{font:13px system-ui,sans-serif;margin:12px;background:#f6f7f9;color:#222}
 h1{font-size:18px;margin:0 0 8px} h2{font-size:15px;margin:4px 0} h3{font-size:13px;margin:8px 0 4px}
 button{font-size:14px;padding:6px 14px;margin-right:6px;cursor:pointer}
 #start{background:#2a7;color:#fff;border:0} #stop{background:#c33;color:#fff;border:0}
 .row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
 .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin:8px 0}
 table{border-collapse:collapse;font-variant-numeric:tabular-nums} td,th{padding:2px 7px;text-align:right;border-bottom:1px solid #eee}
 th{background:#fafafa} td.l,th.l{text-align:left}
 .ok{background:#dff5e1} .warn{background:#fff3c4} .bad{background:#ffd6d6} .muted{color:#888}
 .status{font-weight:600} canvas{background:#fff;border:1px solid #ccc;border-radius:4px;cursor:grab}
 .legend span{display:inline-block;width:14px;height:4px;margin:0 4px 2px 10px;vertical-align:middle}
 .err{color:#b00;white-space:pre-wrap;max-width:600px}
</style></head><body>
<h1>FR3 / GELLO joint inspector <span class="muted" style="font-size:12px">read-only: nothing is commanded</span></h1>
<div>
 <button id="start">Start reading</button><button id="stop">Stop reading</button>
 <span id="run" class="status"></span>
 &nbsp; | &nbsp;<input id="note" size="40" placeholder="note, e.g. 'both at GELLO rest pose'">
 <button id="snap">Save snapshot</button><span id="snapmsg" class="muted"></span>
</div>
<div class="card"><h2>Pairing matrix</h2>
 <div class="muted">largest |GELLO − robot| over J1–J7 (mean in brackets), for every GELLO→robot combination</div>
 <div id="matrix"></div></div>
<div class="card"><h2>Joint limits (deg)</h2><div id="limits"></div></div>
<div id="arms" class="row"></div>
<div id="gellos" class="row"></div>
<script>
const D2R=Math.PI/180, J=[1,2,3,4,5,6,7];
// Franka FR3 kinematics (Craig DH): [a, d, alpha] per joint, then flange d=0.107, hand TCP +0.1034
const DH=[[0,0.333,0],[0,0,-Math.PI/2],[0,0.316,Math.PI/2],[0.0825,0,Math.PI/2],[-0.0825,0.384,-Math.PI/2],[0,0,Math.PI/2],[0.088,0,Math.PI/2]];
let CFG=null, STATE=null, timer=null, pairing={}, views={};
const $=id=>document.getElementById(id);
function mul(A,B){const C=new Array(16).fill(0);for(let r=0;r<4;r++)for(let c=0;c<4;c++)for(let k=0;k<4;k++)C[r*4+c]+=A[r*4+k]*B[k*4+c];return C}
function link(a,d,al,th){const ct=Math.cos(th),st=Math.sin(th),ca=Math.cos(al),sa=Math.sin(al);
 return [ct,-st,0,a, st*ca,ct*ca,-sa,-d*sa, st*sa,ct*sa,ca,d*ca, 0,0,0,1]}
function fk(qdeg){let T=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];const pts=[[0,0,0]];
 for(let i=0;i<7;i++){T=mul(T,link(DH[i][0],DH[i][1],DH[i][2],qdeg[i]*D2R));pts.push([T[3],T[7],T[11]])}
 T=mul(T,link(0,0.107,0,0));pts.push([T[3],T[7],T[11]]);
 const F=T; T=mul(T,link(0,0.1034,0,-Math.PI/4));pts.push([T[3],T[7],T[11]]);
 // flange x / y axis tips (6 cm) to see the wrist roll
 const ax=[F[3]+0.06*F[0],F[7]+0.06*F[4],F[11]+0.06*F[8]], ay=[F[3]+0.06*F[1],F[7]+0.06*F[5],F[11]+0.06*F[9]];
 return {pts,ee:pts[pts.length-1],ax,ay,fl:[F[3],F[7],F[11]]}}
function cls(d){d=Math.abs(d);return d<5?'ok':d<20?'warn':'bad'}
function wrap(d){return ((d+180)%360+360)%360-180}
function f(v,n=1){return v===undefined||v===null?'–':Number(v).toFixed(n)}
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',body:body===undefined?undefined:JSON.stringify(body)});return r.json()}
$('start').onclick=async()=>{await api('/api/start',{});poll()};
$('stop').onclick=async()=>{await api('/api/stop',{});};
$('snap').onclick=async()=>{const r=await api('/api/snapshot',{note:$('note').value,pairing});$('snapmsg').textContent=' saved to '+r.saved_to+' at '+new Date().toLocaleTimeString()};
function dev(k){return STATE&&STATE.devices[k]||{}}
function statusText(d){return d.status==='reading'?'<span style="color:#2a7">reading</span>':d.status==='error'?'<span style="color:#b00">error</span>':(d.status||'idle')}
function buildLayout(){
 const arms=$('arms');arms.innerHTML='';
 for(const name of Object.keys(CFG.robots)){
  const opts=CFG.gellos.map(g=>`<option ${pairing[name]===g?'selected':''}>${g}</option>`).join('');
  arms.insertAdjacentHTML('beforeend',`<div class="card"><h2>Robot ${name} (${CFG.robots[name]}) <span id="st-${name}"></span></h2>
   <div>compare with GELLO <select id="sel-${name}">${opts}</select>
   <span class="legend"><span style="background:#2462c9"></span>robot<span style="background:#e8890c"></span>GELLO</span></div>
   <div class="row"><canvas id="cv-${name}" width="380" height="380" data-view="${name}"></canvas><div><div id="tbl-${name}"></div><div id="info-${name}" class="muted"></div></div></div>
   <div id="err-${name}" class="err"></div></div>`);
  $('sel-'+name).onchange=e=>{pairing[name]=e.target.value;render()};
  orbit('cv-'+name,name);
 }
 const gl=$('gellos');gl.innerHTML='';
 for(const g of CFG.gellos){gl.insertAdjacentHTML('beforeend',`<div class="card"><h2>GELLO ${g} <span id="gst-${g}"></span></h2>
  <div class="legend"><span style="background:#e8890c"></span>GELLO as read (unclipped)<span style="background:#999"></span>after clipping to the limits</div>
  <div class="row"><canvas id="gcv-${g}" width="380" height="380"></canvas><div><div id="gtbl-${g}"></div></div></div><div id="gerr-${g}" class="err"></div></div>`);
  orbit('gcv-'+g,'g:'+g)}
 const LS=CFG.limit_sets;let lt='<table><tr><th class="l">joint</th>'+Object.keys(LS).map(k=>`<th>${k}</th>`).join('')+`<th>active: ${CFG.limits_name}</th></tr>`;
 for(let i=0;i<7;i++)lt+=`<tr><td class="l">J${i+1}</td>`+Object.values(LS).map(s=>`<td>${f(s[i][0])} … ${f(s[i][1])}</td>`).join('')+`<td><b>${f(CFG.limits_deg[i][0])} … ${f(CFG.limits_deg[i][1])}</b></td></tr>`;
 $('limits').innerHTML=lt+'</table><div class="muted">GELLO values and robot commands are clipped to the active set. Near the hardware limits the robot only allows slow motion towards them.</div>';
}
function orbit(id,key){const v=views[key]={yaw:-0.6,pitch:0.35};const cv=$(id);let drag=null;
 cv.onmousedown=e=>drag=[e.clientX,e.clientY,v.yaw,v.pitch];window.addEventListener('mouseup',()=>drag=null);
 cv.onmousemove=e=>{if(!drag)return;v.yaw=drag[2]+(e.clientX-drag[0])*0.01;v.pitch=Math.max(-1.5,Math.min(1.5,drag[3]+(e.clientY-drag[1])*0.01));render()}}
function draw(id,key,layers){
 const cv=$(id),c=cv.getContext('2d'),v=views[key],W=cv.width,H=cv.height,S=300;
 const cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch);
 const P=p=>{const x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];const zz=cp*z+sp*y;return [W/2+S*x,H*0.82-S*zz]};
 c.clearRect(0,0,W,H);c.strokeStyle='#eee';c.lineWidth=1;
 for(let i=-5;i<=5;i++){let a=P([i*0.1,-0.5,0]),b=P([i*0.1,0.5,0]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();
  a=P([-0.5,i*0.1,0]);b=P([0.5,i*0.1,0]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke()}
 const axis=(p,col)=>{c.strokeStyle=col;c.lineWidth=2;c.beginPath();c.moveTo(...P([0,0,0]));c.lineTo(...P(p));c.stroke()};
 axis([0.15,0,0],'#d33');axis([0,0.15,0],'#3a3');axis([0,0,0.15],'#33d');
 c.fillStyle='#999';c.fillText('base x',...P([0.17,0,0]));
 const arm=(q,col,dash)=>{if(!q)return;const k=fk(q);c.setLineDash(dash);c.strokeStyle=col;c.lineWidth=5;c.beginPath();
  k.pts.forEach((p,i)=>{const s=P(p);i?c.lineTo(...s):c.moveTo(...s)});c.stroke();c.setLineDash([]);
  k.pts.slice(1,8).forEach((p,i)=>{const s=P(p);c.fillStyle=col;c.beginPath();c.arc(s[0],s[1],4,0,7);c.fill();c.fillStyle='#000';c.fillText('J'+(i+1),s[0]+5,s[1]-4)});
  c.lineWidth=2;c.strokeStyle='#d33';c.beginPath();c.moveTo(...P(k.fl));c.lineTo(...P(k.ax));c.stroke();
  c.strokeStyle='#3a3';c.beginPath();c.moveTo(...P(k.fl));c.lineTo(...P(k.ay));c.stroke()};
 layers.forEach(l=>arm(l.q,l.col,l.dash||[]));
}
function render(){
 if(!CFG||!STATE)return;
 $('run').innerHTML=STATE.running?'<span style="color:#2a7">● reading</span>':'<span class="muted">stopped</span>';
 const L=CFG.limits_deg;
 // pairing matrix
 const rn=Object.keys(CFG.robots);let m='<table><tr><th class="l">GELLO \\ robot</th>'+rn.map(r=>`<th>${r} (${CFG.robots[r]})</th>`).join('')+'</tr>';
 for(const g of CFG.gellos){m+=`<tr><td class="l">${g}</td>`;for(const r of rn){const a=dev('robot:'+r).q_deg,b=dev('gello:'+g).q_deg;
   if(!a||!b){m+='<td class="muted">–</td>';continue}const d=b.map((x,i)=>Math.abs(x-a[i]));const mx=Math.max(...d),mean=d.reduce((s,x)=>s+x,0)/7;
   m+=`<td class="${cls(mx)}">${f(mx)} (${f(mean)})${pairing[r]===g?' ◀ selected':''}</td>`}m+='</tr>'}
 $('matrix').innerHTML=m+'</table>';
 for(const r of rn){
  const R=dev('robot:'+r),H=dev('hand:'+r),g=pairing[r],G=dev('gello:'+g);
  $('st-'+r).innerHTML=statusText(R)+(R.robot_mode?` · mode <b>${R.robot_mode}</b>`:'');
  $('err-'+r).textContent=[R.error,H.error&&('hand: '+H.error),R.current_errors&&R.current_errors!=='[]'&&R.current_errors!=='No error'?'errors: '+R.current_errors:null].filter(Boolean).join('\n');
  let t='<table><tr><th class="l">joint</th><th>limits</th><th>robot q</th><th>GELLO '+g+'</th><th>GELLO−robot</th><th>τ_ext Nm</th></tr>';
  for(let i=0;i<7;i++){const q=R.q_deg?R.q_deg[i]:null,gq=G.q_deg?G.q_deg[i]:null,d=(q!==null&&gq!==null)?gq-q:null;
   t+=`<tr><td class="l">J${i+1}</td><td class="muted">${f(L[i][0],0)} … ${f(L[i][1],0)}</td><td>${f(q)}</td><td>${f(gq)}</td><td class="${d===null?'':cls(d)}">${f(d)}</td><td>${R.tau_ext_nm?f(R.tau_ext_nm[i],2):'–'}</td></tr>`}
  t+=`<tr><td class="l">hand</td><td></td><td>${H.width_mm!==undefined?H.width_mm+' mm':'–'}</td><td>${G.trigger!==undefined?'trigger '+f(G.trigger,2):'–'}</td><td></td><td></td></tr></table>`;
  $('tbl-'+r).innerHTML=t;
  let info='';
  if(R.q_deg&&G.q_deg){const q=R.q_deg,gq=G.q_deg;
   // same pose, other joint solution: shoulder flip (q1+180, -q2, q3+180) and wrist flip (q5+180, -q6, q7+180)
   const sh=[wrap(gq[0]+180)-q[0],-gq[1]-q[1],wrap(gq[2]+180)-q[2]];
   info+=`if the GELLO shoulder were flipped (J1+180°, −J2, J3+180°): J1–J3 diffs would be ${sh.map(x=>f(x)).join(', ')}<br>`}
  if(R.q_deg&&R.ee_xyz_m){const k=fk(R.q_deg);const e=R.ee_xyz_m;
   info+=`FK check — page EE ${k.ee.map(x=>f(x,3)).join(', ')} m vs robot O_T_EE ${e.map(x=>f(x,3)).join(', ')} m (differs if Desk mounting / hand TCP differ)`}
  $('info-'+r).innerHTML=info;
  draw('cv-'+r,r,[{q:G.q_deg,col:'#e8890c',dash:[8,5]},{q:R.q_deg,col:'#2462c9'}]);
 }
 for(const g of CFG.gellos){const G=dev('gello:'+g);
  $('gst-'+g).innerHTML=statusText(G)+` <span class="muted">${G.port||''}<br>calibration ${G.calibration||'default'}</span>`;
  $('gerr-'+g).textContent=G.error||'';
  let t='<table><tr><th class="l">motor</th><th>ticks</th><th>raw °</th><th>offset °</th><th>sign</th><th>(raw−off)·sign °</th><th>q unclipped °</th><th>q used °</th><th>limits</th></tr>';
  for(let i=0;i<8;i++){
   if(i===7){t+=`<tr><td class="l">8 trigger</td><td>${G.counts?G.counts[7]:'–'}</td><td>${G.raw_deg?f(G.raw_deg[7]):'–'}</td><td colspan="3" class="muted">range ${G.gripper_range_deg?G.gripper_range_deg.join(' … '):'–'}</td><td colspan="3">${f(G.trigger,3)} (0 closed … 1 open)</td></tr>`;continue}
   const qu=G.q_unclipped_deg?G.q_unclipped_deg[i]:null,qq=G.q_deg?G.q_deg[i]:null,clip=qu!==null&&Math.abs(qu-qq)>0.05;
   t+=`<tr><td class="l">${i+1} (J${i+1})</td><td>${G.counts?G.counts[i]:'–'}</td><td>${G.raw_deg?f(G.raw_deg[i]):'–'}</td><td>${G.offsets_deg?f(G.offsets_deg[i]):'–'}</td><td>${G.signs?G.signs[i]:'–'}</td><td>${G.offset_sign_deg?f(G.offset_sign_deg[i]):'–'}</td><td>${f(qu)}</td><td class="${clip?'bad':''}">${f(qq)}${clip?' clipped':''}</td><td class="muted">${f(L[i][0],0)} … ${f(L[i][1],0)}</td></tr>`}
  $('gtbl-'+g).innerHTML=t+'</table>';
  const clipped=G.q_deg&&G.q_unclipped_deg&&G.q_deg.some((x,i)=>Math.abs(x-G.q_unclipped_deg[i])>0.05);
  draw('gcv-'+g,'g:'+g,[clipped?{q:G.q_deg,col:'#999',dash:[4,4]}:{},{q:G.q_unclipped_deg,col:'#e8890c'}])}
}
async function poll(){try{STATE=await api('/api/state');render()}catch(e){$('run').textContent='server not reachable'}
 clearTimeout(timer);timer=setTimeout(poll,STATE&&STATE.running?100:1000)}
(async()=>{CFG=await api('/api/config');pairing={...CFG.pairing};
 for(const r of Object.keys(CFG.robots))if(!pairing[r])pairing[r]=CFG.gellos[0];buildLayout();poll()})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
